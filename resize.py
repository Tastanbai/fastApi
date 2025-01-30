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
