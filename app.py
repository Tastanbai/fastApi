# from fastapi import FastAPI, File, UploadFile, HTTPException
# import face_recognition
# import pickle
# from io import BytesIO
# import time

# # Инициализация FastAPI
# app = FastAPI()

# # Путь к папке с закодированными изображениями
# EMB_PATH = "emb"

# # Список файлов с эмбеддингами
# emb_arr = [
#     "emb.pkl", "emb-0.pkl", "emb-1.pkl", "emb-2.pkl", "emb-3.pkl",
#     "emb-4.pkl", "emb-5.pkl", "emb-6.pkl", "emb-7.pkl", "emb-8.pkl"
# ]

# # Загрузка эмбеддингов в память
# def load_embeddings():
#     embeddings = []
#     for emb_file in emb_arr:
#         with open(f"{EMB_PATH}/{emb_file}", "rb") as file:
#             embeddings.append(pickle.load(file))
#     return embeddings

# # Эмбеддинги загружаются один раз при старте
# known_embeddings = load_embeddings()

# @app.post("/compare-face/")
# async def compare_face(file: UploadFile = File(...)):
#     """
#     API для загрузки фотографии и сравнения с известными эмбеддингами.
#     """
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="Загруженный файл не является изображением.")

#     try:
#         # Сравнение с известными эмбеддингами
#         start_time = time.time()

#         # Загрузка изображения из запроса
#         content = await file.read()
#         unknown_image = face_recognition.load_image_file(BytesIO(content))

#         # Получение эмбеддинга загруженного изображения
#         unknown_face_encodings = face_recognition.face_encodings(unknown_image)
#         if not unknown_face_encodings:
#             raise HTTPException(status_code=400, detail="Лицо на изображении не найдено.")
        
#         unknown_face_encoding = unknown_face_encodings[0]

#         results = []
#         for known_encoding in known_embeddings:
#             distance = face_recognition.face_distance([known_encoding], unknown_face_encoding)[0]
#             similarity_percentage = (1 - distance) * 100
#             results.append(similarity_percentage)
        
#         # Находим максимальное сходство
#         max_similarity = max(results)
#         end_time = time.time()

#         return {
#             "similarity_percentage": f"{max_similarity:.2f}%",
#             "execution_time": f"{end_time - start_time:.2f} seconds"
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from datetime import datetime
# import os
# import time
# from face_save import extract_and_save_faces
# from resize import resize_image
# from save_emb import save_embedding_to_file
# import psycopg
# from psycopg.rows import dict_row

# # Инициализация FastAPI
# app = FastAPI()

# # Папки для сохранения данных
# OUTPUT_FOLDER = "faces_folder"
# EMB_FOLDER = "emb"
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# os.makedirs(EMB_FOLDER, exist_ok=True)

# # Настройки подключения к PostgreSQL
# DB_HOST = "localhost"
# DB_USER = "postgres"
# DB_PASSWORD = "123456"
# DB_NAME = "face_db"
# DB_PORT = 5433

# # Функция для сохранения данных в базу PostgreSQL
# def save_to_database(patient_id, hospital_id, branch_id, palata_id, image_path, emb_path):
#     try:
#         with psycopg.connect(
#             host=DB_HOST,
#             dbname=DB_NAME,
#             user=DB_USER,
#             password=DB_PASSWORD,
#             port=DB_PORT,
#             row_factory=dict_row
#         ) as conn:
#             with conn.cursor() as cur:
#                 sql = """
#                 INSERT INTO faces (patient_id, hospital_id, branch_id, palata_id, image_path, emb_path)
#                 VALUES (%s, %s, %s, %s, %s, %s)
#                 """
#                 cur.execute(sql, (patient_id, hospital_id, branch_id, palata_id, image_path, emb_path))
#                 conn.commit()
#     except Exception as e:
#         raise Exception(f"Ошибка при сохранении данных в базу: {e}")

# # Эндпоинт для загрузки данных пациента
# @app.post("/process-patient/")
# async def process_patient(
#     patient_id: int = Form(...),
#     hospital_id: int = Form(...),
#     branch_id: int = Form(...),
#     palata_id: int = Form(...),
#     file: UploadFile = File(...)
# ):
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="Загруженный файл не является изображением.")

#     try:
#         # Сохранение исходного изображения
#         content = await file.read()
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         original_path = os.path.join(OUTPUT_FOLDER, f"original_{timestamp}.jpg").replace("\\", "/")
#         with open(original_path, "wb") as f:
#             f.write(content)
#         print(f"Оригинальное изображение сохранено: {original_path}")

#         # Извлечение лица
#         face_save_path = extract_and_save_faces(original_path, OUTPUT_FOLDER)

#         if not face_save_path or not os.path.exists(face_save_path):
#             raise ValueError(f"Файл {face_save_path} не найден после извлечения лица.")

#         # Изменение размера изображения
#         resized_path = os.path.join(OUTPUT_FOLDER, f"resized_{timestamp}.jpg").replace("\\", "/")
#         resize_image(face_save_path, resized_path)

#         if not os.path.exists(resized_path):
#             raise ValueError(f"Файл {resized_path} не найден после изменения размера.")

#         # Создание эмбеддинга
#         start_time = time.time()
#         emb_path = os.path.join(EMB_FOLDER, f"emb_{timestamp}.pkl").replace("\\", "/")
#         save_embedding_to_file(resized_path, emb_path)
#         embedding_time = time.time() - start_time

#         if not os.path.exists(emb_path):
#             raise ValueError(f"Файл {emb_path} не найден после создания эмбеддинга.")

#         # Сохранение данных в базу
#         save_to_database(patient_id, hospital_id, branch_id, palata_id, resized_path, emb_path)

#         return {
#             "message": "Данные пациента успешно обработаны.",
#             "resized_image_path": resized_path,
#             "embedding_path": emb_path,
#             "embedding_time": f"{embedding_time:.2f} seconds"
#         }

#     except Exception as e:
#         print(f"Ошибка: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

#/////////////////////////////////////////////////////////////////////////////////


# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from datetime import datetime
# import os
# import time
# from face_save import extract_and_save_faces
# from resize import resize_image
# from save_emb import save_embedding_to_file
# import psycopg
# from psycopg.rows import dict_row
# import face_recognition
# import pickle
# from io import BytesIO

# # Инициализация FastAPI
# app = FastAPI()

# # Папки для сохранения данных
# OUTPUT_FOLDER = "faces_folder"
# EMB_FOLDER = "emb"
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# os.makedirs(EMB_FOLDER, exist_ok=True)

# # Настройки подключения к PostgreSQL
# DB_HOST = "localhost"
# DB_USER = "postgres"
# DB_PASSWORD = "123456"
# DB_NAME = "face_db"
# DB_PORT = 5433

# # Функция для сохранения данных в базу PostgreSQL
# def save_to_database(patient_id, hospital_id, branch_id, palata_id, image_path, emb_path):
#     try:
#         with psycopg.connect(
#             host=DB_HOST,
#             dbname=DB_NAME,
#             user=DB_USER,
#             password=DB_PASSWORD,
#             port=DB_PORT,
#             row_factory=dict_row
#         ) as conn:
#             with conn.cursor() as cur:
#                 sql = """
#                 INSERT INTO faces (patient_id, hospital_id, branch_id, palata_id, image_path, emb_path)
#                 VALUES (%s, %s, %s, %s, %s, %s)
#                 """
#                 cur.execute(sql, (patient_id, hospital_id, branch_id, palata_id, image_path, emb_path))
#                 conn.commit()
#     except Exception as e:
#         raise Exception(f"Ошибка при сохранении данных в базу: {e}")

# # Загрузка всех эмбеддингов в память
# def load_embeddings():
#     embeddings = []
#     for emb_file in os.listdir(EMB_FOLDER):
#         emb_path = os.path.join(EMB_FOLDER, emb_file)
#         with open(emb_path, "rb") as file:
#             embeddings.append(pickle.load(file))
#     return embeddings

# known_embeddings = load_embeddings()

# # Эндпоинт для обработки пациента
# @app.post("/process-patient/")
# async def process_patient(
#     patient_id: int = Form(...),
#     hospital_id: int = Form(...),
#     branch_id: int = Form(...),
#     palata_id: int = Form(...),
#     file: UploadFile = File(...)
# ):
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="Загруженный файл не является изображением.")

#     try:
#         # Сохранение исходного изображения
#         content = await file.read()
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         original_path = os.path.join(OUTPUT_FOLDER, f"original_{timestamp}.jpg").replace("\\", "/")
#         with open(original_path, "wb") as f:
#             f.write(content)
#         print(f"Оригинальное изображение сохранено: {original_path}")

#         # Извлечение лица
#         face_save_path = extract_and_save_faces(original_path, OUTPUT_FOLDER)

#         if not face_save_path or not os.path.exists(face_save_path):
#             raise ValueError(f"Файл {face_save_path} не найден после извлечения лица.")

#         # Изменение размера изображения
#         resized_path = os.path.join(OUTPUT_FOLDER, f"resized_{timestamp}.jpg").replace("\\", "/")
#         resize_image(face_save_path, resized_path)

#         if not os.path.exists(resized_path):
#             raise ValueError(f"Файл {resized_path} не найден после изменения размера.")

#         # Создание эмбеддинга
#         start_time = time.time()
#         emb_path = os.path.join(EMB_FOLDER, f"emb_{timestamp}.pkl").replace("\\", "/")
#         save_embedding_to_file(resized_path, emb_path)
#         embedding_time = time.time() - start_time

#         if not os.path.exists(emb_path):
#             raise ValueError(f"Файл {emb_path} не найден после создания эмбеддинга.")

#         # Сохранение данных в базу
#         save_to_database(patient_id, hospital_id, branch_id, palata_id, resized_path, emb_path)

#         return {
#             "message": "Данные пациента успешно обработаны.",
#             "resized_image_path": resized_path,
#             "embedding_path": emb_path,
#             "embedding_time": f"{embedding_time:.2f} seconds"
#         }

#     except Exception as e:
#         print(f"Ошибка: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# # Эндпоинт для сравнения лица
# @app.post("/compare-face/")
# async def compare_face(file: UploadFile = File(...)):
#     """
#     API для загрузки фотографии и сравнения с известными эмбеддингами.
#     """
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="Загруженный файл не является изображением.")

#     try:
#         # Сравнение с известными эмбеддингами
#         start_time = time.time()

#         # Загрузка изображения из запроса
#         content = await file.read()
#         unknown_image = face_recognition.load_image_file(BytesIO(content))

#         # Получение эмбеддинга загруженного изображения
#         unknown_face_encodings = face_recognition.face_encodings(unknown_image)
#         if not unknown_face_encodings:
#             raise HTTPException(status_code=400, detail="Лицо на изображении не найдено.")
        
#         unknown_face_encoding = unknown_face_encodings[0]

#         results = []
#         comparison_start = time.time()
#         for known_encoding in known_embeddings:
#             distance = face_recognition.face_distance([known_encoding], unknown_face_encoding)[0]
#             similarity_percentage = (1 - distance) * 100
#             results.append(similarity_percentage)
#         comparison_time = time.time() - comparison_start
        
#         # Находим максимальное сходство
#         max_similarity = max(results) if results else 0.0
#         end_time = time.time()

#         return {
#             "similarity_percentage": f"{max_similarity:.2f}%",
#             "execution_time": f"{end_time - start_time:.2f} seconds",
#             "comparison_time": f"{comparison_time:.2f} seconds"
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))





from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from datetime import datetime
import os
import time
import numpy as np
import cv2
import face_recognition
import pickle
from face_save import extract_and_save_faces
from resize import resize_image
from save_emb import save_embedding_to_file
import psycopg
from psycopg.rows import dict_row

# Инициализация FastAPI
app = FastAPI()

# Папки для сохранения данных
OUTPUT_FOLDER = "faces_folder"
EMB_FOLDER = "emb"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(EMB_FOLDER, exist_ok=True)

# Настройки подключения к PostgreSQL
DB_HOST = "localhost"
DB_USER = "postgres"
DB_PASSWORD = "123456"
DB_NAME = "face_db"
DB_PORT = 5433

# Функция для сохранения данных в базу PostgreSQL
def save_to_database(patient_id, hospital_id, branch_id, palata_id, image_path, emb_path):
    try:
        with psycopg.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            row_factory=dict_row
        ) as conn:
            with conn.cursor() as cur:
                sql = """
                INSERT INTO faces (patient_id, hospital_id, branch_id, palata_id, image_path, emb_path)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cur.execute(sql, (patient_id, hospital_id, branch_id, palata_id, image_path, emb_path))
                conn.commit()
    except Exception as e:
        raise Exception(f"Ошибка при сохранении данных в базу: {e}")

# Загрузка всех эмбеддингов
def load_all_embeddings():
    embeddings = []
    try:
        for emb_file in os.listdir(EMB_FOLDER):
            emb_path = os.path.join(EMB_FOLDER, emb_file)
            with open(emb_path, "rb") as file:
                embeddings.append(pickle.load(file))
    except Exception as e:
        raise Exception(f"Ошибка при загрузке эмбеддингов: {e}")
    return embeddings

known_embeddings = load_all_embeddings()

@app.post("/process-patient/")
async def process_patient(
    patient_id: int = Form(...),
    hospital_id: int = Form(...),
    branch_id: int = Form(...),
    palata_id: int = Form(...),
    file: UploadFile = File(...)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Загруженный файл не является изображением.")

    try:
        # Сохранение исходного изображения
        content = await file.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_path = os.path.join(OUTPUT_FOLDER, f"original_{timestamp}.jpg").replace("\\", "/")
        with open(original_path, "wb") as f:
            f.write(content)
        print(f"Оригинальное изображение сохранено: {original_path}")

        # Извлечение лица
        face_save_path = extract_and_save_faces(original_path, OUTPUT_FOLDER)

        if not face_save_path or not os.path.exists(face_save_path):
            raise ValueError(f"Файл {face_save_path} не найден после извлечения лица.")

        # Изменение размера изображения
        resized_path = os.path.join(OUTPUT_FOLDER, f"resized_{timestamp}.jpg").replace("\\", "/")
        resize_image(face_save_path, resized_path)

        if not os.path.exists(resized_path):
            raise ValueError(f"Файл {resized_path} не найден после изменения размера.")

        # Создание эмбеддинга
        start_time = time.time()
        emb_path = os.path.join(EMB_FOLDER, f"emb_{timestamp}.pkl").replace("\\", "/")
        save_embedding_to_file(resized_path, emb_path, num_jitters=10)  # Повышение точности
        embedding_time = time.time() - start_time

        if not os.path.exists(emb_path):
            raise ValueError(f"Файл {emb_path} не найден после создания эмбеддинга.")

        # Сохранение данных в базу
        save_to_database(patient_id, hospital_id, branch_id, palata_id, resized_path, emb_path)

        return {
            "message": "Данные пациента успешно обработаны.",
            "resized_image_path": resized_path,
            "embedding_path": emb_path,
            "embedding_time": f"{embedding_time:.2f} seconds"
        }

    except Exception as e:
        print(f"Ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-face/")
async def compare_face(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Загруженный файл не является изображением.")

    try:
        # Загрузка изображения
        content = await file.read()
        unknown_image = face_recognition.load_image_file(BytesIO(content))

        # Получение эмбеддинга загруженного изображения
        start_time = time.time()
        unknown_face_encodings = face_recognition.face_encodings(unknown_image, num_jitters=3)
        if not unknown_face_encodings:
            raise HTTPException(status_code=400, detail="Лицо на изображении не найдено.")
        unknown_face_encoding = unknown_face_encodings[0]
        embedding_time = time.time() - start_time

        # Сравнение с известными эмбеддингами
        results = []
        comparison_start = time.time()
        for known_encoding in known_embeddings:
            distance = face_recognition.face_distance([known_encoding], unknown_face_encoding)[0]
            similarity_percentage = (1 - distance) * 100
            results.append(similarity_percentage)
       
        max_similarity = max(results) if results else 0.0

        return {
            "similarity_percentage": f"{max_similarity:.2f}%",
            "embedding_time": f"{embedding_time:.2f} seconds",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
