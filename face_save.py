import cv2
import dlib
import os
from datetime import datetime

def extract_and_save_faces(image_path, output_folder, face_margin=20):
    print(f"Извлечение лица из файла: {image_path}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение. Проверьте путь: {image_path}")

    detector = dlib.get_frontal_face_detector()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)

    print(f"Найдено лиц: {len(faces)}")

    if len(faces) == 0:
        raise ValueError("Лицо не обнаружено на изображении.")

    for i, face in enumerate(faces):
        x1 = max(face.left() - face_margin, 0)
        y1 = max(face.top() - face_margin, 0)
        x2 = min(face.right() + face_margin, image.shape[1])
        y2 = min(face.bottom() + face_margin, image.shape[0])

        face_image = image[y1:y2, x1:x2]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = os.path.join(output_folder, f"face_{timestamp}.jpg").replace("\\", "/")
        cv2.imwrite(output_path, face_image)

        print(f"Файл лица сохранён: {output_path}")

        if not os.path.exists(output_path):
            raise ValueError(f"Файл {output_path} не был создан.")

        return output_path  # Возвращаем путь сохранённого лица
