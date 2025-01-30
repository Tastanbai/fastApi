import face_recognition
import pickle

def save_embedding_to_file(image_path, output_path, num_jitters=1):
    """
    Создаёт эмбеддинг лица из изображения и сохраняет его в файл.
    
    :param image_path: Путь к изображению.
    :param output_path: Путь для сохранения эмбеддинга.
    :param num_jitters: Количество искажений для повышения точности (по умолчанию 1).
    """
    # Загрузка изображения
    image = face_recognition.load_image_file(image_path)

    # Получение эмбеддинга
    face_encodings = face_recognition.face_encodings(image, num_jitters=num_jitters)
    if not face_encodings:
        raise ValueError(f"Лицо не обнаружено в изображении: {image_path}")

    face_encoding = face_encodings[0]

    # Сохранение эмбеддинга
    with open(output_path, "wb") as file:
        pickle.dump(face_encoding, file)

    print(f"Эмбеддинг сохранён в {output_path}")

