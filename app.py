import numpy as np
import base64
import mysql.connector
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Query
import time
from datetime import datetime
import os
import face_recognition
import pickle
from resize import resize_image
from face_save import extract_and_save_faces
from save_emb import save_embedding_to_file
from auth import router as auth_router, get_current_user
from register import router as register_router
from fastapi import Request
import json
from typing import List, Optional

app = FastAPI()

app.include_router(auth_router)
app.include_router(register_router)

OUTPUT_FOLDER = "faces_folder"
EMB_FOLDER = "emb"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(EMB_FOLDER, exist_ok=True)

DB_CONFIG = {
    "host": "face.tabet-kitap.kz",
    "user": "fastapi_user",
    "password": "secure_password",
    "database": "face_db",
    "port": 3306
}


async def check_permission(user: dict, request):
    """ Проверяет доступ пользователя к API """
    
    allowed_api_str = user.get("allowed_api", "[]")  # Получаем строку
    try:
        allowed_api = json.loads(allowed_api_str)  # Конвертируем JSON в список
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Ошибка в формате данных API")

    api_name = request.scope["path"].strip("/")  

    if "*" in allowed_api or api_name in allowed_api:
        return  

    raise HTTPException(status_code=403, detail=f"У вас нет доступа к API {api_name}")



# ✅ Глобальные переменные
known_embeddings = []
known_patients = []
known_hospitals = []  # Добавляем hospital_id



# ✅ Функция загрузки эмбеддингов, patient_id и hospital_id
def load_embeddings_from_database():
    global known_embeddings, known_patients, known_hospitals
    embeddings, patients, hospitals = [], [], []

    try:
        with mysql.connector.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT patient_id, hospital_id, emb_path FROM faces")
                results = cursor.fetchall()

        for row in results:
            patient_id, hospital_id, emb_file = row
            if os.path.exists(emb_file):
                with open(emb_file, "rb") as file:
                    embeddings.append(pickle.load(file))
                    patients.append(patient_id)
                    hospitals.append(hospital_id) 
            else:
                print(f"⚠️ Файл {emb_file} отсутствует, пропускаем...")

    except mysql.connector.Error as e:
        print(f"❌ Ошибка при загрузке эмбеддингов из БД: {e}")

    # Обновляем глобальные переменные
    known_embeddings[:] = embeddings
    known_patients[:] = patients
    known_hospitals[:] = hospitals  

# 🔥 Загружаем эмбеддинги при старте сервера
load_embeddings_from_database()




# 📌 API для добавления пациента и АВТОМАТИЧЕСКОГО ОБНОВЛЕНИЯ эмбеддингов
@app.post("/process-patient/")
async def process_patient(
    request: Request,
    patient_id: str = Form(...),
    hospital_id: str = Form(...),
    branch_id: str = Form(...),
    palata_id: str = Form(...),
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    ):

    await check_permission(user, request)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    emb_folder = "emb"

    # 📌 Сохранение изображения
    original_path = f"faces_folder/original_{timestamp}.jpg"
    content = await file.read()
    with open(original_path, "wb") as f:
        f.write(content)

    # 📌 Обработка лица
    face_image = face_recognition.load_image_file(original_path)
    face_encodings = face_recognition.face_encodings(face_image)

    if not face_encodings:
        raise HTTPException(status_code=400, detail="Лицо не найдено.")
    
    start_time = time.time()
    embedding = face_encodings[0]
    emb_path = os.path.join(emb_folder, f"emb_{timestamp}.pkl")

    # 📌 Сохранение эмбеддинга
    with open(emb_path, "wb") as f:
        pickle.dump(embedding, f)
    embedding_time = time.time() - start_time  # Конец замера времени
    
    # 📌 Сохранение в базу
    try:
        with mysql.connector.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                sql = """
                INSERT INTO faces (patient_id, hospital_id, branch_id, palata_id, image_path, emb_path)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (patient_id, hospital_id, branch_id, palata_id, original_path, emb_path))
                conn.commit()
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении в базу: {e}")

    # 📌 АВТОМАТИЧЕСКОЕ ОБНОВЛЕНИЕ ЭМБЕДДИНГОВ (загружаем новые)
    load_embeddings_from_database()

    return {
        "message": "Пациент добавлен и эмбеддинги обновлены!",
        "image_path": original_path,
        "embedding_path": emb_path,
        "embedding_time": f"{embedding_time:.2f} seconds",
    }

@app.post("/compare-face/")
async def compare_face(
    file: UploadFile = File(...),
):

    content = await file.read()
    unknown_image = face_recognition.load_image_file(BytesIO(content))
    start_time = time.time()
    unknown_face_encodings = face_recognition.face_encodings(unknown_image)

    if not unknown_face_encodings:
        raise HTTPException(status_code=400, detail="Лицо не найдено.")

    unknown_face_encoding = unknown_face_encodings[0]

    if not known_embeddings:
        raise HTTPException(status_code=400, detail="Нет загруженных эмбеддингов.")

    # ✅ Сравнение с известными лицами
    distances = face_recognition.face_distance(known_embeddings, unknown_face_encoding)
    min_distance = min(distances, default=1.0)
    similarity_percentage = (1 - min_distance) * 100
    comparison_time = time.time() - start_time

    # ✅ Поиск ближайшего patient_id и hospital_id
    matched_patient_id = None
    matched_hospital_id = None
    status = False

    if similarity_percentage >= 65.0:
        index = distances.argmin()
        matched_patient_id = known_patients[index]
        matched_hospital_id = known_hospitals[index]  

        status = True

    # ✅ Сохранение результата в MySQL
    try:
        with mysql.connector.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO faceData (patient_id, hospital_id, status, similarity_percentage, comparison_time, timestamp)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    """,
                    (matched_patient_id, matched_hospital_id, status, similarity_percentage, comparison_time)
                )
                conn.commit()
    except mysql.connector.Error as e:
        print(f"❌ Ошибка при записи в БД: {e}")

    return {
        "status": status,
        "patient_id": matched_patient_id,
        "hospital_id": matched_hospital_id,  # ✅ Добавляем hospital_id в ответ
        "similarity_percentage": float(similarity_percentage),
        "comparison_time": f"{comparison_time:.2f} seconds"
    }

@app.post("/process-patient-base64/")
async def process_patient_base64(
    request: Request,
    patient_id: str = Form(...),
    hospital_id: str = Form(...),
    branch_id: str = Form(...),
    palata_id: str = Form(...),
    file_base64: str = Form(...),
    user: dict = Depends(get_current_user),
    ):

    await check_permission(user, request)
    
    """
    Эндпоинт для приёма base64-изображения вместо UploadFile
    """
    try:
        image_data = base64.b64decode(file_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Невозможно декодировать base64: {e}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    try:
        # 📌 Сохранение оригинального изображения
        original_path = os.path.join(OUTPUT_FOLDER, f"original_{timestamp}.jpg")
        with open(original_path, "wb") as f:
            f.write(image_data)

        # 📌 Обработка лица
        face_save_path = extract_and_save_faces(original_path, OUTPUT_FOLDER)
        resized_path = os.path.join(OUTPUT_FOLDER, f"resized_{timestamp}.jpg")
        resize_image(face_save_path, resized_path)

        # 📌 Создание эмбеддинга
        start_time = time.time()
        emb_path = os.path.join(EMB_FOLDER, f"emb_{timestamp}.pkl")
        save_embedding_to_file(resized_path, emb_path)
        embedding_time = time.time() - start_time  # Конец замера времени

        # 📌 Сохранение в базу данных
        try:
            with mysql.connector.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cursor:
                    sql = """
                    INSERT INTO faces (patient_id, hospital_id, branch_id, palata_id, image_path, emb_path)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (patient_id, hospital_id, branch_id, palata_id, resized_path, emb_path))
                    conn.commit()
        except mysql.connector.Error as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при сохранении в БД: {e}")

        # 🔥 АВТОМАТИЧЕСКОЕ ОБНОВЛЕНИЕ ЭМБЕДДИНГОВ (загружаем новые)
        load_embeddings_from_database()

        return {
            "message": "Данные пациента успешно обработаны (base64) и эмбеддинги обновлены!",
            "resized_image_path": resized_path,
            "embedding_path": emb_path,
            "embedding_time": f"{embedding_time:.2f} seconds"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare-face-numpy/")
async def compare_face(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    ):
    
    await check_permission(user, request)

    total_start_time = time.time()  # 🕒 Начало замера времени всей функции

    content = await file.read()
    unknown_image = face_recognition.load_image_file(BytesIO(content))

    # 🔹 Уменьшаем размер изображения для быстрого обнаружения лиц
    unknown_image = np.array(unknown_image)
    small_unknown_image = np.array(unknown_image[::2, ::2, :])  # Уменьшаем в 2 раза

    # 🔹 Извлекаем эмбеддинг
    # 🕒 Начало замера времени только для сравнения
    comparison_start_time = time.time()
    unknown_face_encodings = face_recognition.face_encodings(small_unknown_image)

    if not unknown_face_encodings:
        raise HTTPException(status_code=400, detail="Лицо не найдено.")

    unknown_face_encoding = unknown_face_encodings[0]

    if not known_embeddings:
        raise HTTPException(status_code=400, detail="Нет загруженных эмбеддингов для сравнения.")

    # 🔹 Преобразуем список эмбеддингов в NumPy массив (ускоряет вычисления)
    known_embeddings_array = np.array(known_embeddings, dtype=np.float32)

    # 🔥 Сравнение лиц без перебора в Python (ускоряет в 2-3 раза)
    distances = face_recognition.face_distance(known_embeddings_array, unknown_face_encoding)
    
    min_distance = np.min(distances, initial=1.0)
    similarity_percentage = (1 - min_distance) * 100
    comparison_time = time.time() - comparison_start_time  # Конец замера времени сравнения

    total_execution_time = time.time() - total_start_time  # Полное время выполнения

    return {
        "status": bool(similarity_percentage >= 55.0),
        "similarity_percentage": float(similarity_percentage),
        "comparison_time": f"{comparison_time:.2f} seconds",
        "total_execution_time": f"{total_execution_time:.2f} seconds"  # Полное время выполнения
    }

# 🔹 API для сравнения лиц с базой
@app.post("/compare-face-qr/")
async def compare_face_with_db(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    ):
    
    await check_permission(user, request)

    total_start_time = time.time()

    # Проверяем, загружены ли эмбеддинги
    if not known_embeddings or not known_patients:
        raise HTTPException(status_code=500, detail="База эмбеддингов не загружена.")

    # Загружаем фото и уменьшаем размер
    content = await file.read()
    unknown_image = face_recognition.load_image_file(BytesIO(content))
    small_unknown_image = np.array(unknown_image[::2, ::2, :])

    # Извлекаем эмбеддинг загруженного лица
    comparison_start_time = time.time()
    unknown_face_encodings = face_recognition.face_encodings(small_unknown_image)

    if not unknown_face_encodings:
        raise HTTPException(status_code=400, detail="Лицо не найдено.")

    unknown_face_encoding = unknown_face_encodings[0]

    # Преобразуем список эмбеддингов в NumPy массив
    known_embeddings_array = np.array(known_embeddings, dtype=np.float32)

    # Сравнение лиц
    distances = face_recognition.face_distance(known_embeddings_array, unknown_face_encoding)
    min_index = np.argmin(distances)  # Индекс минимального расстояния
    min_distance = distances[min_index]

    similarity_percentage = (1 - min_distance) * 100
    status = bool(similarity_percentage >= 55.0)
    patient_id = known_patients[min_index] if status else None

    comparison_time = time.time() - comparison_start_time
    total_execution_time = time.time() - total_start_time

    # Если лицо распознано, записываем в БД
    if status:
        try:
            with mysql.connector.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cursor:
                    insert_query = "INSERT INTO QR (status, patient_id) VALUES (%s, %s)"
                    cursor.execute(insert_query, (status, patient_id))
                    conn.commit()
        except mysql.connector.Error as e:
            print(f"❌ Ошибка записи в БД: {e}")

    return {
        "status": status,
        "patient_id": patient_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if status else None,
    }


# ✅ API для получения данных с фильтрацией по дате, hospital_id и patient_id
@app.get("/get-face-data/")
async def get_face_data(
    request: Request,
    start_date: Optional[str] = Query(None, description="Начальная дата (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Конечная дата (YYYY-MM-DD)"),
    hospital_id: Optional[str] = Query(None, description="ID больницы"),
    patient_id: Optional[str] = Query(None, description="ID пациента"),
    user: dict = Depends(get_current_user),
):
    await check_permission(user, request)

    # ✅ Проверяем, что даты в правильном формате (если переданы)
    def validate_date(date_str):
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Неверный формат даты: {date_str}. Используйте YYYY-MM-DD")

    if start_date:
        start_date = validate_date(start_date)
    if end_date:
        end_date = validate_date(end_date)

    try:
        with mysql.connector.connect(**DB_CONFIG) as conn:
            with conn.cursor(dictionary=True) as cursor:
                query = """
                SELECT id, patient_id, hospital_id, status, similarity_percentage, comparison_time, timestamp 
                FROM faceData
                """
                params = []

                # ✅ Фильтрация по дате, hospital_id и patient_id
                conditions = []
                if start_date and end_date:
                    conditions.append("timestamp BETWEEN %s AND %s")
                    params.extend([start_date + " 00:00:00", end_date + " 23:59:59"])
                elif start_date:
                    conditions.append("timestamp >= %s")
                    params.append(start_date + " 00:00:00")
                elif end_date:
                    conditions.append("timestamp <= %s")
                    params.append(end_date + " 23:59:59")

                if hospital_id:
                    conditions.append("hospital_id = %s")
                    params.append(hospital_id)

                if patient_id:
                    conditions.append("patient_id = %s")
                    params.append(patient_id)

                # ✅ Добавляем WHERE только если есть условия
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY timestamp DESC"

                cursor.execute(query, params)
                results = cursor.fetchall()

        return {"count": len(results), "data": results}

    except mysql.connector.Error as e:
        return {"error": f"Ошибка при получении данных: {e}"}



# 📌 API для сравнения лиц
# @app.post("/compare-face/")
# async def compare_face(
#     request: Request,
#     file: UploadFile = File(...),
#     user: dict = Depends(get_current_user),
#     ):
    
#     await check_permission(user, request)
    
#     content = await file.read()
#     unknown_image = face_recognition.load_image_file(BytesIO(content))
#     start_time = time.time()
#     unknown_face_encodings = face_recognition.face_encodings(unknown_image)

#     if not unknown_face_encodings:
#         raise HTTPException(status_code=400, detail="Лицо не найдено.")
    
   
#     unknown_face_encoding = unknown_face_encodings[0]

#     if not known_embeddings:
#         raise HTTPException(status_code=400, detail="Нет загруженных эмбеддингов для сравнения.")

#     # 🔥 Сравнение лиц без чтения файлов/базы (максимальная скорость)
   
#     distances = face_recognition.face_distance(known_embeddings, unknown_face_encoding)
#     min_distance = min(distances, default=1.0)
#     similarity_percentage = (1 - min_distance) * 100
#     comparison_time = time.time() - start_time

#     return {
#         "status": bool(similarity_percentage >= 55.0),
#         "similarity_percentage": float(similarity_percentage),
#         "comparison_time": f"{comparison_time:.2f} seconds"
#     }


# @app.post("/compare-face/")
# async def compare_face(
#     # request: Request,
#     file: UploadFile = File(...),
#     # user: dict = Depends(get_current_user),
# ):
#     # await check_permission(user, request)

#     content = await file.read()
#     unknown_image = face_recognition.load_image_file(BytesIO(content))
#     start_time = time.time()
#     unknown_face_encodings = face_recognition.face_encodings(unknown_image)

#     if not unknown_face_encodings:
#         raise HTTPException(status_code=400, detail="Лицо не найдено.")

#     unknown_face_encoding = unknown_face_encodings[0]

#     if not known_embeddings:
#         raise HTTPException(status_code=400, detail="Нет загруженных эмбеддингов.")

#     # ✅ Сравнение с известными лицами
#     distances = face_recognition.face_distance(known_embeddings, unknown_face_encoding)
#     min_distance = min(distances, default=1.0)
#     similarity_percentage = (1 - min_distance) * 100
#     comparison_time = time.time() - start_time

#     # ✅ Поиск ближайшего patient_id
#     matched_patient_id = None
#     status = False
#     if similarity_percentage >= 55.0:
#         index = distances.argmin()  
#         matched_patient_id = known_patients[index]
#         status = True

#     # ✅ Сохранение результата в MySQL
#     try:
#         with mysql.connector.connect(**DB_CONFIG) as conn:
#             with conn.cursor() as cursor:
#                 cursor.execute(
#                     """
#                     INSERT INTO faceData (patient_id, status, similarity_percentage, comparison_time, timestamp)
#                     VALUES (%s, %s, %s, %s, NOW())
#                     """,
#                     (matched_patient_id, status, similarity_percentage, comparison_time)
#                 )
#                 conn.commit()
#     except mysql.connector.Error as e:
#         print(f"❌ Ошибка при записи в БД: {e}")

#     return {
#         "status": status,
#         "patient_id": matched_patient_id,
#         "similarity_percentage": float(similarity_percentage),
#         "comparison_time": f"{comparison_time:.2f} seconds"
#     }

