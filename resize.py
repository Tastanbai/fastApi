# import cv2

# def resize_image(image_path, output_path, target_width=400):
#     """
#     Уменьшает изображение до указанной ширины (по умолчанию 400 пикселей), сохраняя соотношение сторон.
    
#     :param image_path: Путь к исходному изображению.
#     :param output_path: Путь для сохранения уменьшенного изображения.
#     :param target_width: Желаемая ширина изображения (по умолчанию 400 пикселей).
#     """
#     # Загрузка изображения
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Не удалось загрузить изображение. Проверьте путь к файлу.")

#     # Получение текущих размеров изображения
#     height, width = image.shape[:2]

#     # Проверка, нужно ли уменьшать изображение
#     if width > target_width:
#         # Вычисление нового размера
#         scale_ratio = target_width / width
#         new_width = target_width
#         new_height = int(height * scale_ratio)

#         # Изменение размера изображения
#         resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

#         # Сохранение уменьшенного изображения
#         cv2.imwrite(output_path, resized_image)
#         print(f"Изображение сохранено в {output_path} с размером {new_width}x{new_height}.")
#     else:
#         print("Размер изображения меньше целевой ширины. Изменение размера не требуется.")


# #resize_image("input.jpg", "output.jpg", target_width=400)
# # Пример использования
# resize_image("files/face_today.jpg", "faces_folder/face_today.jpg", target_width=400)

import cv2

import cv2
import shutil


def resize_image(image_path, output_path, target_width=400):
    """
    Уменьшает изображение до указанной ширины, сохраняя пропорции.
    Если исходная ширина <= target_width, файл просто копируется,
    что быстрее, чем повторно декодировать/кодировать изображение.
    """
    # Считываем изображение (для определения размеров)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение. Проверьте путь: {image_path}")

    height, width = image.shape[:2]

    if width > target_width:
        # Выполняем ресайз только при необходимости
        scale_ratio = target_width / width
        new_width = target_width
        new_height = int(height * scale_ratio)

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, resized_image)

        print(f"Изображение сохранено в {output_path} с размером {new_width}x{new_height}.")
    else:
        # Если ресайз не нужен, копируем файл для скорости
        shutil.copy2(image_path, output_path)
        print(
            f"Исходное изображение (ширина {width}px) меньше или равно "
            f"целевой ({target_width}px). Файл просто скопирован в {output_path}."
        )
