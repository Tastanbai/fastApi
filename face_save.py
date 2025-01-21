# import cv2
# import dlib
# import os
# from datetime import datetime


# def extract_and_save_faces(image_path, output_folder, face_margin=20):
#     """
#     Извлекает лица из изображения и сохраняет их в указанной папке.
    
#     :param image_path: Путь к исходному изображению.
#     :param output_folder: Папка для сохранения изображений лиц.
#     :param face_margin: Дополнительный отступ вокруг лица (в пикселях).
#     """
#     # Убедиться, что выходная папка существует
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Загрузка изображения
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Не удалось загрузить изображение. Проверьте путь к файлу.")

#     # Инициализация детектора лиц dlib
#     detector = dlib.get_frontal_face_detector()

#     # Конвертация изображения в оттенки серого (требуется для dlib)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Обнаружение лиц
#     faces = detector(gray_image)

#     if len(faces) == 0:
#         print("Лица не обнаружены на изображении.")
#         return

#     # Итерация по всем обнаруженным лицам
#     for i, face in enumerate(faces):
#         x1 = max(face.left() - face_margin, 0)
#         y1 = max(face.top() - face_margin, 0)
#         x2 = min(face.right() + face_margin, image.shape[1])
#         y2 = min(face.bottom() + face_margin, image.shape[0])

#         # Вырезать лицо из изображения
#         face_image = image[y1:y2, x1:x2]

#         # Уменьшить размер лица до ширины 300 пикселей, если оно больше
#         face_height, face_width = face_image.shape[:2]
#         if face_width > 300:
#             scale_ratio = 300 / face_width
#             new_width = 300
#             new_height = int(face_height * scale_ratio)
#             face_image = cv2.resize(face_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

#         # Генерация уникального имени файла по локальной дате и времени
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#         output_path = os.path.join(output_folder, f"face_{timestamp}.jpg")

#         # Сохранение лица в выходную папку
#         cv2.imwrite(output_path, face_image)
#         print(f"Лицо сохранено: {output_path}")


# # Пример использования
# extract_and_save_faces("files/facetau.jpg", "faces_folder", face_margin=20)

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
