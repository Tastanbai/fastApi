import numpy as np
import base64
import mysql.connector
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
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

# Глобальная переменная для эмбеддингов
known_embeddings = []
known_patients = []

# 🔥 Функция загрузки эмбеддингов и patient_id из БД
def load_embeddings_from_database():
    global known_embeddings, known_patients
    embeddings = []
    patients = []

    try:
        with mysql.connector.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT patient_id, emb_path FROM faces")
                results = cursor.fetchall()

        for row in results:
            patient_id, emb_file = row
            if os.path.exists(emb_file):
                with open(emb_file, "rb") as file:
                    embeddings.append(pickle.load(file))
                    patients.append(patient_id)
            else:
                print(f"⚠️ Файл {emb_file} отсутствует, пропускаем...")

    except mysql.connector.Error as e:
        print(f"❌ Ошибка при загрузке эмбеддингов из БД: {e}")

    known_embeddings = embeddings  # Обновляем глобальный список эмбеддингов
    known_patients = patients  # Обновляем список patient_id

# 🔥 Загружаем эмбеддинги при старте сервера
load_embeddings_from_database()

# # Функция загрузки эмбеддингов (вызывается при каждом новом пациенте)
# def load_embeddings_from_database():
#     global known_embeddings
#     embeddings = []
#     try:
#         with mysql.connector.connect(**DB_CONFIG) as conn:
#             with conn.cursor() as cursor:
#                 cursor.execute("SELECT emb_path FROM faces")
#                 results = cursor.fetchall()

#         for row in results:
#             emb_file = row[0]
#             if os.path.exists(emb_file):
#                 with open(emb_file, "rb") as file:
#                     embeddings.append(pickle.load(file))
#             else:
#                 print(f"⚠️ Файл {emb_file} отсутствует, пропускаем...")
#     except mysql.connector.Error as e:
#         print(f"❌ Ошибка при загрузке эмбеддингов из БД: {e}")
    
#     known_embeddings = embeddings  # Обновляем глобальный список

# # 🔥 Загружаем эмбеддинги при старте сервера
# load_embeddings_from_database()

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

# 📌 API для сравнения лиц
@app.post("/compare-face/")
async def compare_face(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    ):
    
    await check_permission(user, request)
    
    content = await file.read()
    unknown_image = face_recognition.load_image_file(BytesIO(content))
    start_time = time.time()
    unknown_face_encodings = face_recognition.face_encodings(unknown_image)

    if not unknown_face_encodings:
        raise HTTPException(status_code=400, detail="Лицо не найдено.")
    
   
    unknown_face_encoding = unknown_face_encodings[0]

    if not known_embeddings:
        raise HTTPException(status_code=400, detail="Нет загруженных эмбеддингов для сравнения.")

    # 🔥 Сравнение лиц без чтения файлов/базы (максимальная скорость)
   
    distances = face_recognition.face_distance(known_embeddings, unknown_face_encoding)
    min_distance = min(distances, default=1.0)
    similarity_percentage = (1 - min_distance) * 100
    comparison_time = time.time() - start_time

    return {
        "status": bool(similarity_percentage >= 55.0),
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
    
    check_permission("process_patient_base64", user)  # Проверка доступа
    
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