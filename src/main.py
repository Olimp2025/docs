import cv2
import numpy as np


class TableExtractor:

    def __init__(self, dict_path, file_name):
        self.image_path = dict_path + "/" + file_name

    def execute(self):
        """
            Функция выравнивает изображение и обрезает края

            type_of_crop:
                "all_tables" - обрезает изображение по краям и оставляет все таблицы
                "only_amount" - обрезает изображение по краям таблицы по выполненным работам
        """
        self.read_image()
        self.store_process_image("0_original.jpg", self.image)
        self.convert_image_to_grayscale()
        self.store_process_image("1_grayscaled.jpg", self.grayscale_image)
        self.threshold_image()
        self.store_process_image("3_thresholded.jpg", self.thresholded_image)
        self.invert_image()
        self.store_process_image("4_inverteded.jpg", self.inverted_image)
        self.dilate_image()
        self.store_process_image("5_dialateded.jpg", self.dilated_image)
        self.edge_detection()
        self.store_process_image("6_edges.jpg", self.dilated_image)
        self.image_rotation()
        self.store_process_image("7_rotated.jpg", self.dilated_image)
        self.save_image()

        return

    def read_image(self):
        self.image = cv2.imread(self.image_path)

    def convert_image_to_grayscale(self):
        self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def blur_image(self):
        self.blurred_image = cv2.blur(self.grayscale_image, (5, 5))

    def threshold_image(self):
        self.thresholded_image = cv2.threshold(self.grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def invert_image(self):
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)

    def dilate_image(self):
        self.dilated_image = cv2.dilate(self.inverted_image, None, iterations=10)

    def edge_detection(self):
        # Применяем Canny для выделения границ
        self.edges = cv2.Canny(self.dilated_image, 50, 150, apertureSize=3)

        # Находим контуры
        contours, _ = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Находим максимальную длину самой длинной стороны
        # Фильтруем контуры, оставляя только те, у которых длина самой длинной стороны ≥ 70% от max_length
        # Сортируем оставшиеся контуры по длине самой длинной стороны в убывающем порядке
        max_length = max(max(cv2.minAreaRect(c)[1]) for c in contours)
        filtered_contours = [c for c in contours if max(cv2.minAreaRect(c)[1]) >= 0.7 * max_length]
        contours = sorted(filtered_contours, key=lambda c: max(cv2.minAreaRect(c)[1]), reverse=True)

        # Находим контуры с самой длинной стороной
        #contours = sorted(contours, key=lambda c: max(cv2.minAreaRect(c)[1]), reverse=True)[:5]

        # Проверяем, найдены ли контуры
        if not contours:
            print("Контуры не найдены")
            return

        # Определяем углы наклона всех контуров
        angles = []
        rotated_rects = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            if angle < -45:
                angle += 90  # Коррекция угла
            angles.append(angle)
            rotated_rects.append((rect, cnt))

        # Определяем медианный угол наклона (чтобы исключить выбросы)
        median_angle = np.median(angles)

        # Фильтруем контуры, удаляя те, у которых угол слишком отличается
        threshold = 3  # Допустимое отклонение в градусах
        self.filtered_contours = [cnt for rect, cnt in rotated_rects if abs(rect[-1] - median_angle) <= threshold]

        # Отбираем самые длинные горизонтальные объекты
        filtered_rects = sorted([rect for rect, cnt in rotated_rects if abs(rect[-1] - median_angle) <= threshold],
                                key=lambda rect: rect[1][0], reverse=True)

        # Копируем изображение для отображения результатов
        self.image_with_horizontal_contours = self.image.copy()

        # Рисуем повернутые прямоугольники
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

        print("Контуры найдены")

    def image_rotation(self) -> None:
        """ Поворачивает исходное изображение на медианный угол, чтобы его выровнять """
        if self.median_filtered_angle is None:
            print("Ошибка: угол не определен.")
            return

        # Коррекция угла: OpenCV поворачивает в противоположном направлении
        angle = self.median_filtered_angle

        # Определяем центр изображения
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)

        # Создаем матрицу поворота
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Определяем новые размеры изображения после поворота
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Корректируем матрицу поворота, чтобы учесть смещение центра
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2

        # Применяем поворот с учетом новых размеров
        self.rotated_image = cv2.warpAffine(self.image, rotation_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)

        print("Поворот")

    def save_image(self) -> None:
        """ Сохраняет повернутое изображение, если оно было создано """
        if hasattr(self, 'rotated_image') and self.rotated_image is not None:
            cv2.imwrite(f"corrected_images/{file_name}", self.rotated_image)
            print("Финальное повернутое изображение сохранено.")
        else:
            print("Ошибка: повернутое изображение отсутствует.")

    def store_process_image(self, file_name, image):
        path = "./process_images/table_extractor/" + file_name
        cv2.imwrite(path, image)


if __name__ == '__main__':
    dict_path = "target_images"
    file_name = "02_rotated_1.jpg"
    TE = TableExtractor(dict_path, file_name)
    TE.execute()

