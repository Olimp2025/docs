import cv2
import numpy as np
import os
import random

input_folder = "input_images/"
output_folder_images = "target_images/"
output_folder_angles = "rotation_angles/"

os.makedirs(output_folder_images, exist_ok=True)
os.makedirs(output_folder_angles, exist_ok=True)


# Функция поворота изображения без обрезки краев
def rotate_images(image, angle):
    (h, w) = image.shape[:2]

    # Вычисляем новый размер изображения после поворота
    diagonal = int(np.sqrt(h ** 2 + w ** 2))
    extended_size = (diagonal, diagonal)

    # Создаем расширенный холст с белым фоном
    extended_image = np.ones((diagonal, diagonal, 3), dtype=np.uint8) * 255
    center_x, center_y = diagonal // 2, diagonal // 2
    start_x, start_y = (diagonal - w) // 2, (diagonal - h) // 2

    extended_image[start_y:start_y + h, start_x:start_x + w] = image
    center = (center_x, center_y)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(extended_image, M, extended_size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


if __name__ == "__main__":
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg", ".tif", ".bmp")):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Ошибка загрузки: {filename}")
                continue

            # Создаем 4 повернутые версии изображения
            for i in range(4):
                angle = random.uniform(-40, 40)
                rotated_image = rotate_images(image, angle)

                # Создаем уникальное имя файла с номером варианта
                base_name, ext = os.path.splitext(filename)
                output_image_name = f"{base_name}_rotated_{i + 1}{ext}"
                output_image_path = os.path.join(output_folder_images, output_image_name)

                cv2.imwrite(output_image_path, rotated_image)

                # Сохраняем угол поворота в текстовый файл
                angle_file_path = os.path.join(output_folder_angles, f"{base_name}_rotated_{i + 1}.txt")
                with open(angle_file_path, "w") as f:
                    f.write(f"{angle:.2f}")

                print(f"Обработано: {filename} -> {output_image_name}, угол поворота: {angle:.2f} градусов")