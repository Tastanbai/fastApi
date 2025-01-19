import time
import face_recognition
import pickle

emb_arr = ["emb.pkl","emb-0.pkl","emb-1.pkl","emb-2.pkl","emb-3.pkl","emb-4.pkl","emb-5.pkl","emb-6.pkl","emb-7.pkl","emb-8.pkl"] 

# Запись времени начала
start_time = time.time()

# Загрузка неизвестного изображения
unknown_image = face_recognition.load_image_file("faces_folder/face_1.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

for x in emb_arr:
    with open(f"emb/{x}", "rb") as file:
        loaded_face_encoding = pickle.load(file)
    distance = face_recognition.face_distance([loaded_face_encoding], unknown_face_encoding)[0]

similarity_percentage = (1 - distance) * 100

# Запись времени окончания
end_time = time.time()

print(f"Сходство: {similarity_percentage:.2f}%, Время выполнения: {end_time - start_time:.2f} секунд")