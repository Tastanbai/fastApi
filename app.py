from fastapi import FastAPI, File, UploadFile, HTTPException
import face_recognition
import pickle
from io import BytesIO
import time

# Инициализация FastAPI
app = FastAPI()

# Путь к папке с закодированными изображениями
EMB_PATH = "emb"

# Список файлов с эмбеддингами
emb_arr = [
    "emb.pkl", "emb-0.pkl", "emb-1.pkl", "emb-2.pkl", "emb-3.pkl",
    "emb-4.pkl", "emb-5.pkl", "emb-6.pkl", "emb-7.pkl", "emb-8.pkl"
]

# Загрузка эмбеддингов в память
def load_embeddings():
    embeddings = []
    for emb_file in emb_arr:
        with open(f"{EMB_PATH}/{emb_file}", "rb") as file:
            embeddings.append(pickle.load(file))
    return embeddings

# Эмбеддинги загружаются один раз при старте
known_embeddings = load_embeddings()

@app.post("/compare-face/")
async def compare_face(file: UploadFile = File(...)):
    """
    API для загрузки фотографии и сравнения с известными эмбеддингами.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Загруженный файл не является изображением.")

    try:
        # Сравнение с известными эмбеддингами
        start_time = time.time()

        # Загрузка изображения из запроса
        content = await file.read()
        unknown_image = face_recognition.load_image_file(BytesIO(content))

        # Получение эмбеддинга загруженного изображения
        unknown_face_encodings = face_recognition.face_encodings(unknown_image)
        if not unknown_face_encodings:
            raise HTTPException(status_code=400, detail="Лицо на изображении не найдено.")
        
        unknown_face_encoding = unknown_face_encodings[0]

        results = []
        for known_encoding in known_embeddings:
            distance = face_recognition.face_distance([known_encoding], unknown_face_encoding)[0]
            similarity_percentage = (1 - distance) * 100
            results.append(similarity_percentage)
        
        # Находим максимальное сходство
        max_similarity = max(results)
        end_time = time.time()

        return {
            "similarity_percentage": f"{max_similarity:.2f}%",
            "execution_time": f"{end_time - start_time:.2f} seconds"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))