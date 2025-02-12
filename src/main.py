import cv2
import numpy as np


class TableExtractor:

    def __init__(self, dict_path, file_name):
        self.image_path = dict_path + "/" + file_name

    def execute(self):
        """
            Класс выравнивает изображение
        """
        self.read_image()
        self.filter_image()
        self.edge_detection()
        self.image_rotation()
        self.save_image()

        return

    def read_image(self):
        self.image = cv2.imread(self.image_path)

    def filter_image(self):
        self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.thresholded_image = cv2.threshold(self.grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)
        self.dilated_image = cv2.dilate(self.inverted_image, None, iterations=10)

    def edge_detection(self):
        # Применяем Canny для выделения границ
        # Находим контуры
        # Находим максимальную длину самой длинной стороны
        # Фильтруем контуры, оставляя только те, у которых длина самой длинной стороны ≥ 70% от max_length
        # Сортируем оставшиеся контуры по длине самой длинной стороны в убывающем порядке

        self.edges = cv2.Canny(self.dilated_image, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_length = max(max(cv2.minAreaRect(c)[1]) for c in contours)
        filtered_contours = [c for c in contours if max(cv2.minAreaRect(c)[1]) >= 0.7 * max_length]
        contours = sorted(filtered_contours, key=lambda c: max(cv2.minAreaRect(c)[1]), reverse=True)

        # Проверяем, найдены ли контуры
        if not contours:
            print("Контуры не найдены")
            return

        # Определяем углы наклона всех контуров
        # Определяем медианный угол наклона (чтобы исключить выбросы)
        angles = []
        rotated_rects = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            if angle < -45:
                angle += 90  # Коррекция угла
            angles.append(angle)
            rotated_rects.append((rect, cnt))
        median_angle = np.median(angles)

        # Фильтруем контуры, удаляя те, у которых угол слишком отличается
        # Отбираем самые длинные горизонтальные объекты
        threshold = 3  # Допустимое отклонение в градусах
        self.filtered_contours = [cnt for rect, cnt in rotated_rects if abs(rect[-1] - median_angle) <= threshold]
        filtered_rects = sorted([rect for rect, cnt in rotated_rects if abs(rect[-1] - median_angle) <= threshold],
                                key=lambda rect: rect[1][0], reverse=True)

        # Копируем изображение для отображения результатов
        # Рисуем повернутые прямоугольники
        self.image_with_horizontal_contours = self.image.copy()
        for rect in filtered_rects:
            box = cv2.boxPoints(rect)  # Получаем углы прямоугольника
            box = np.intp(box)  # Преобразуем координаты в целые числа
            cv2.drawContours(self.image_with_horizontal_contours, [box], 0, (0, 255, 0),
                             3)  # Рисуем повернутый прямоугольник

        # Определяем медианный угол среди отфильтрованных контуров
        if self.filtered_contours:
            self.median_filtered_angle = 0
            self.median_filtered_angle = np.median([cv2.minAreaRect(cnt)[-1] for cnt in self.filtered_contours])
        else:
            self.median_filtered_angle = None
            print("Нет отфильтрованных контуров для вычисления медианного угла")

        print("Контуры")

    def image_rotation(self) -> None:
        """ Поворачивает исходное изображение на медианный угол и сохраняет оригинальное разрешение """
        if self.median_filtered_angle is None:
            print("Ошибка: угол не определен.")
            return

        # Коррекция угла
        # Определяем центр и размеры изображения
        # Создаем матрицу поворота
        # Применяем поворот с оригинальными размерами
        angle = self.median_filtered_angle
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_full = cv2.warpAffine(self.image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)

        # Центрируем и обрезаем изображение до исходного размера
        crop_x = (rotated_full.shape[1] - w) // 2
        crop_y = (rotated_full.shape[0] - h) // 2
        self.rotated_image = rotated_full[crop_y:crop_y + h, crop_x:crop_x + w]

        print("Поворот выполнен и размер сохранен")

    def save_image(self) -> None:
        """ Сохраняет повернутое изображение, если оно было создано """
        if hasattr(self, 'rotated_image') and self.rotated_image is not None:
            cv2.imwrite(f"corrected_images/{file_name}", self.rotated_image)
            print("Финальное повернутое изображение сохранено.")
        else:
            print("Ошибка: повернутое изображение отсутствует.")


if __name__ == '__main__':
    dict_path = "target_images"
    file_name = "02_rotated_1.jpg"
    TE = TableExtractor(dict_path, file_name)
    TE.execute()

