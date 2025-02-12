import cv2
import numpy as np
import os


class DocRotater:
    def __init__(self, dict_path: str, file_name: str, output_dir: str) -> None:
        """
        Инициализация класса с указанием директории и имени файла.
        """
        self.dict_path = dict_path
        self.file_name = file_name
        self.image_path = os.path.join(dict_path, file_name)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def execute(self) -> None:
        """
        Выравнивает изображение: чтение, фильтрация, детекция контуров, поворот и сохранение.
        """
        if not self.read_image():
            print("Ошибка при чтении изображения.")
            return

        self.filter_image()
        self.edge_detection()
        self.image_rotation()
        self.save_image()

    def read_image(self) -> bool:
        """
        Читает изображение с диска.
        Возвращает True, если изображение успешно загружено, иначе False.
        """
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print(f"Ошибка: не удалось прочитать изображение по пути {self.image_path}")
            return False
        return True

    def filter_image(self) -> None:
        """
        Применяет базовую фильтрацию: преобразование в оттенки серого, пороговую бинаризацию,
        инверсию и дилатацию.
        """
        self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.thresholded_image = cv2.threshold(
            self.grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)
        self.dilated_image = cv2.dilate(self.inverted_image, None, iterations=10)

    def edge_detection(self) -> None:
        """
        Применяет Canny для выделения границ, находит контуры, фильтрует их по длине
        и вычисляет медианный угол наклона для выравнивания.
        """
        self.edges = cv2.Canny(self.dilated_image, 50, 150, apertureSize=3)
        # cv2.findContours возвращает разные значения в зависимости от версии OpenCV
        contours_info = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

        if not contours:
            print("Контуры не найдены")
            self.median_filtered_angle = None
            return

        # Вычисляем максимальную длину стороны среди всех контуров
        try:
            max_length = max(max(cv2.minAreaRect(c)[1]) for c in contours)
        except ValueError:
            print("Ошибка при вычислении максимальной длины контура.")
            self.median_filtered_angle = None
            return

        # Фильтруем контуры по условию: длина стороны >= 70% от максимальной
        filtered_contours = [c for c in contours if max(cv2.minAreaRect(c)[1]) >= 0.7 * max_length]
        contours_sorted = sorted(filtered_contours, key=lambda c: max(cv2.minAreaRect(c)[1]), reverse=True)

        if not contours_sorted:
            print("Нет подходящих контуров после фильтрации.")
            self.median_filtered_angle = None
            return

        # Вычисляем углы наклона для каждого контура с коррекцией
        angles = []
        rotated_rects = []
        for cnt in contours_sorted:
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            if angle < -45:
                angle += 90  # Коррекция угла
            angles.append(angle)
            rotated_rects.append((rect, cnt))
        median_angle = np.median(angles)

        # Фильтрация контуров по отклонению угла от медианного
        threshold = 3  # Допустимое отклонение в градусах
        self.filtered_contours = [
            cnt for rect, cnt in rotated_rects if abs(rect[-1] - median_angle) <= threshold
        ]

        # Для визуализации (если необходимо) – рисуем повернутые прямоугольники
        self.image_with_horizontal_contours = self.image.copy()
        filtered_rects = sorted(
            [rect for rect, cnt in rotated_rects if abs(rect[-1] - median_angle) <= threshold],
            key=lambda rect: rect[1][0],
            reverse=True
        )
        for rect in filtered_rects:
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(self.image_with_horizontal_contours, [box], 0, (0, 255, 0), 3)

        if self.filtered_contours:
            angles = []
            for cnt in self.filtered_contours:
                angle = cv2.minAreaRect(cnt)[-1]
                if angle < -45:  # Коррекция угла
                    angle += 90
                angles.append(angle)

            self.median_filtered_angle = np.median(angles) if angles else None
        else:
            self.median_filtered_angle = None
            print("Нет отфильтрованных контуров для вычисления медианного угла")

        print("Контуры обработаны, медианный угол:", self.median_filtered_angle)

    def image_rotation(self) -> None:
        """
        Поворачивает исходное изображение на медианный угол.
        Выходное изображение сохраняет исходное разрешение.
        """
        if self.median_filtered_angle is None:
            print("Ошибка: угол не определен.")
            return

        if 90 > self.median_filtered_angle > 45:
            angle = self.median_filtered_angle - 90
        else:
            angle = self.median_filtered_angle
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Применяем поворот с оригинальными размерами (w, h)
        self.rotated_image = cv2.warpAffine(
            self.image, rotation_matrix, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        print("Поворот выполнен и размер сохранен")

    def save_image(self) -> None:
        """
        Сохраняет повернутое изображение в папке output_dir с тем же именем, что и исходный файл.
        """
        if hasattr(self, 'rotated_image') and self.rotated_image is not None:
            output_path = os.path.join(self.output_dir, self.file_name)
            cv2.imwrite(output_path, self.rotated_image)
            print("Финальное повернутое изображение сохранено:", output_path)
        else:
            print("Ошибка: повернутое изображение отсутствует.")


if __name__ == '__main__':
    dict_path = "target_images"
    output_dir = "corrected_images"
    os.makedirs(output_dir, exist_ok=True)

    # Перебираем все файлы в директории dict_path
    for file_name in os.listdir(dict_path):
        if file_name.startswith("."):
            continue

        print(f"Обрабатывается файл: {file_name}")
        extractor = DocRotater(dict_path, file_name, output_dir)
        extractor.execute()

    print("Обработка всех файлов завершена.")