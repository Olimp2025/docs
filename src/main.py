import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

import src.utils


class DocRotater:
    MIN_CONTOUR_RATIO = 0.7  # Минимальное соотношение для фильтрации контуров по длине стороны
    ANGLE_THRESHOLD = 3      # Допустимое отклонение угла в градусах

    def __init__(self, dict_path: str, file_name: str, output_dir: str) -> None:
        """
        Пути к файлам и выходной директории
        """
        self.dict_path = dict_path
        self.file_name = file_name
        self.image_path = os.path.join(dict_path, file_name)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.image = None
        self.gray = None
        self.thresh = None
        self.inverted = None
        self.dilated = None
        self.edges = None
        self.filtered_contours = None
        self.median_angle = None
        self.rotated_image = None
        self.image_with_horizontal_contours = None

    def execute(self) -> None:
        """
        Выравнивает изображение: чтение, фильтрация, детекция контуров, поворот и сохранение.
        """
        if not self.read_image():
            print("Ошибка при чтении изображения.")
            return

        self.filter_image()
        self.edge_detection()  # Сравнение углов после детекции
        self.image_rotation()
        self.save_image()

    def read_image(self) -> bool:
        """
        Читает изображение с диска.
        """
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print(f"Ошибка: не удалось прочитать изображение по пути {self.image_path}")
            return False
        return True

    def filter_image(self) -> None:
        """
        Применяет базовую фильтрацию
        """
        if self.image is None:
            raise ValueError("Изображение не загружено.")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, self.thresh = cv2.threshold(
            self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        self.inverted = cv2.bitwise_not(self.thresh)
        # Используем явно заданное ядро для дилатации
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.dilated = cv2.dilate(self.inverted, kernel, iterations=10)

    @src.utils.compare_angle
    def edge_detection(self) -> None:
        """
        Применяет Canny для выделения границ, находит контуры, фильтрует их по длине
        и вычисляет медианный угол наклона для выравнивания.
        """
        if self.dilated is None:
            raise ValueError("Дилатированное изображение не определено.")
        self.edges = cv2.Canny(self.dilated, 50, 150, apertureSize=3)
        contours_info = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

        if not contours:
            print("Контуры не найдены")
            self.median_angle = None
            return

        # Вычисляем минимальные прямоугольники и максимальную сторону для каждого контура
        rects = []
        max_side_list = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            max_side = max(rect[1])
            rects.append((rect, cnt, max_side))
            max_side_list.append(max_side)
        max_length = max(max_side_list)

        # Фильтрация по условию MIN_CONTOUR_RATIO
        rects_filtered = [(rect, cnt) for rect, cnt, side in rects if side >= self.MIN_CONTOUR_RATIO * max_length]
        if not rects_filtered:
            print("Нет подходящих контуров после фильтрации.")
            self.median_angle = None
            return

        # Сортировка по максимальной стороне (в убывающем порядке)
        rects_filtered.sort(key=lambda x: max(x[0][1]), reverse=True)

        # Вычисляем корректированный угол для каждого прямоугольника
        angles = []
        rects_with_angle = []
        for rect, cnt in rects_filtered:
            angle = rect[-1]
            if angle < -45:
                angle += 90  # Коррекция угла
            angles.append(angle)
            rects_with_angle.append((rect, cnt, angle))

        median_angle = np.median(angles)
        self.median_angle = median_angle

        # Фильтрация контуров по отклонению угла от медианного
        self.filtered_contours = [
            cnt for rect, cnt, angle in rects_with_angle if abs(angle - median_angle) <= self.ANGLE_THRESHOLD
        ]

        # Визуализация: рисуем прямоугольники с отклонением меньше порога
        self.image_with_horizontal_contours = self.image.copy()
        for rect, _, angle in rects_with_angle:
            if abs(angle - median_angle) <= self.ANGLE_THRESHOLD:
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(self.image_with_horizontal_contours, [box], 0, (0, 255, 0), 3)

        print("Контуры обработаны, медианный угол:", self.median_angle)

    def image_rotation(self) -> None:
        """
        Поворачивает исходное изображение на вычисленный медианный угол.
        """
        if self.median_angle is None:
            print("Ошибка: угол не определен.")
            return

        # Корректировка угла, если он лежит в диапазоне (45, 90)
        angle = self.median_angle
        if 90 > angle > 45:
            angle -= 90

        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.rotated_image = cv2.warpAffine(
            self.image, M, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        print("Поворот выполнен и размер сохранен")

    def save_image(self) -> None:
        """
        Сохраняет повернутое изображение в выходной директории.
        """
        if self.rotated_image is not None:
            output_path = os.path.join(self.output_dir, self.file_name)
            cv2.imwrite(output_path, self.rotated_image)
            print("Финальное повернутое изображение сохранено:", output_path)
        else:
            print("Ошибка: повернутое изображение отсутствует.")


def process_file(file_name, dict_path, output_dir):
    """
    Функция для обработки отдельного файла.
    """
    if file_name.startswith("."):
        return

    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in allowed_extensions:
        return

    print(f"\nОбрабатывается файл: {file_name}")
    try:
        extractor = DocRotater(dict_path, file_name, output_dir)
        extractor.execute()
    except Exception as e:
        print(f"Ошибка при обработке файла {file_name}: {e}")


if __name__ == '__main__':
    dict_path = "output_images"
    output_dir = "corrected_images"
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(dict_path) if not f.startswith(".")]

    # Параллельная обработка файлов для ускорения работы
    with ThreadPoolExecutor() as executor:
        for file_name in files:
            executor.submit(process_file, file_name, dict_path, output_dir)

    # Вывод итоговых результатов
    print("\nОбщие результаты:")
    print(f"Общее количество совпавших углов: {src.utils.total_matched}")
    print(f"Общее количество несовпавших углов: {src.utils.total_unmatched}")
    print(f"Общее количество нераспознанных углов: {src.utils.total_not_compared}")

    total = src.utils.total_matched + src.utils.total_unmatched + src.utils.total_not_compared
    Accuracy = src.utils.total_matched / total if total > 0 else 0
    print(f"Средняя точность: {Accuracy:.2f}")