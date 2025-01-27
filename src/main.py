import cv2
import numpy as np


def deskew_image(input_path, output_path):
    # Шаг 1: Загрузка изображения
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить файл {input_path}")

    # Шаг 2: Преобразование в градиент серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Шаг 3: (Опционально) Инверсия изображения
    # Это полезно, когда текст тёмный на светлом фоне.
    # Если текст у вас уже чёрный на белом, можно пропустить этот шаг
    gray = cv2.bitwise_not(gray)

    # Шаг 4: Бинаризация (Otsu)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Шаг 5: Нахождение всех координат белых пикселей
    coords = np.column_stack(np.where(thresh > 0))

    # Если вдруг документ почти пустой, контролируем ситуацию
    if len(coords) == 0:
        # Сохраняем исходное изображение без изменений и завершаем
        cv2.imwrite(output_path, image)
        print("Не удалось найти объекты для расчёта угла. Файл сохранён без изменений.")
        return

    # Используем метод minAreaRect для вычисления угла
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # minAreaRect может возвращать угол от -90 до 0, нужно поправить его
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Шаг 6: Поворот изображения на найденный угол
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Матрица поворота
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Аффинное преобразование (сглаживаем интерполяцией INTER_CUBIC,
    # границы восполняем "replicate", чтобы не было чёрных углов)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    # Шаг 7: Сохранение результата
    cv2.imwrite(output_path, rotated)

    # Выводим угол поворота
    print(f"Найденный угол поворота: {angle:.2f} град.")


# Пример использования:
if __name__ == "__main__":
    input_file = "scan_06.jpg"
    output_file = "scan_06_corrected.jpg"

    deskew_image(input_file, output_file)