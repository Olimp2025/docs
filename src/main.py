import cv2
import numpy as np


class TableExtractor:

    def __init__(self, image_path):
        self.image_path = image_path

    def execute(self, type_of_crop):
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
        self.store_process_image("5_2_dialateded.jpg", self.dilated_image)

        self.find_contours()
        self.store_process_image("6_all_contours.jpg", self.image_with_all_contours)
        self.filter_contours_and_leave_only_rectangles()
        self.store_process_image("7_only_rectangular_contours.jpg", self.image_with_only_rectangular_contours)
        self.find_largest_contour_by_area()
        self.store_process_image("8_contour_with_max_area.jpg", self.image_with_contour_with_max_area)
        self.order_points_in_the_contour_with_max_area()
        self.store_process_image("9_with_4_corner_points_plotted.jpg", self.image_with_points_plotted)
        self.calculate_new_width_and_height_of_image()
        if type_of_crop == "all_tables":
            self.apply_perspective_transform_all_tables()
        if type_of_crop == "only_amount":
            self.apply_perspective_transform()
        self.store_process_image("10_perspective_corrected.jpg", self.perspective_corrected_image)
        # self.add_10_percent_padding()
        # self.store_process_image("11_perspective_corrected_with_padding.jpg",
        #                          self.perspective_corrected_image_with_padding)
        # self.crop()
        return self.perspective_corrected_image, self.contour_with_max_area

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
        self.dilated_image = cv2.dilate(self.inverted_image, None, iterations=13)

    def edge_detection(self):
        # Применяем Canny для выделения границ
        self.edges = cv2.Canny(self.dilated_image, 50, 150, apertureSize=3)
        # Находим контуры
        contours, _ = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Проверяем, найдены ли контуры
        if not contours:
            print("Контуры не найдены")
            return
        # Отбираем самые длинные горизонтальные объекты
        horizontal_contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[2], reverse=True)[:5]
        # Копируем изображение для отображения результатов
        self.image_with_horizontal_contours = self.image.copy()
        # Рисуем повернутые прямоугольники
        for cnt in horizontal_contours:
            rect = cv2.minAreaRect(cnt)  # Получаем минимальный ограничивающий прямоугольник
            box = cv2.boxPoints(rect)  # Получаем углы прямоугольника
            box = np.intp(box)  # Преобразуем координаты в целые числа
            cv2.drawContours(self.image_with_horizontal_contours, [box], 0, (0, 255, 0),
                             3)  # Рисуем повернутый прямоугольник
        print("Контуры найдены")

    def find_contours(self):
        self.contours, self.hierarchy = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.image_with_all_contours = self.image.copy()
        cv2.drawContours(self.image_with_all_contours, self.contours, -1, (0, 255, 0), 3)

    def filter_contours_and_leave_only_rectangles(self):
        self.rectangular_contours = []
        for contour in self.contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                self.rectangular_contours.append(approx)
        self.image_with_only_rectangular_contours = self.image.copy()
        cv2.drawContours(self.image_with_only_rectangular_contours, self.rectangular_contours, -1, (0, 255, 0), 3)

    def find_largest_contour_by_area(self):
        max_area = 0
        self.contour_with_max_area = None
        for contour in self.rectangular_contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                self.contour_with_max_area = contour
        self.image_with_contour_with_max_area = self.image.copy()
        cv2.drawContours(self.image_with_contour_with_max_area, [self.contour_with_max_area], -1, (0, 255, 0), 3)

    def order_points_in_the_contour_with_max_area(self):
        self.contour_with_max_area_ordered = self.order_points(self.contour_with_max_area)
        self.image_with_points_plotted = self.image.copy()
        for point in self.contour_with_max_area_ordered:
            point_coordinates = (int(point[0]), int(point[1]))
            self.image_with_points_plotted = cv2.circle(self.image_with_points_plotted, point_coordinates, 10,
                                                        (0, 0, 255), -1)

    def calculate_new_width_and_height_of_image(self):
        existing_image_width = self.image.shape[1]
        existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)

        distance_between_top_left_and_top_right = self.calculateDistanceBetween2Points(
            self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[1])
        distance_between_top_left_and_bottom_left = self.calculateDistanceBetween2Points(
            self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[3])

        aspect_ratio = distance_between_top_left_and_bottom_left / distance_between_top_left_and_top_right

        self.new_image_width = existing_image_width_reduced_by_10_percent
        self.new_image_height = int(self.new_image_width * aspect_ratio)

    def apply_perspective_transform(self):
        pts1 = np.float32(self.contour_with_max_area_ordered)
        pts2 = np.float32([[0, 0], [self.new_image_width, 0], [self.new_image_width, self.new_image_height],
                           [0, self.new_image_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.perspective_corrected_image = cv2.warpPerspective(self.image, matrix,
                                                               (self.new_image_width, self.new_image_height))

    def apply_perspective_transform_all_tables(self):
        pts1 = np.float32(self.contour_with_max_area_ordered)
        pts2 = np.float32([[0, 420], [self.new_image_width, 420], [self.new_image_width, 3271],
                           [0, 3271]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.perspective_corrected_image = cv2.warpPerspective(self.image, matrix,
                                                               (self.new_image_width, 3485))

    def add_10_percent_padding(self):
        image_height = self.image.shape[0]
        padding = int(image_height * 0.1)
        self.perspective_corrected_image_with_padding = cv2.copyMakeBorder(self.perspective_corrected_image, padding,
                                                                           padding, padding, padding,
                                                                           cv2.BORDER_CONSTANT, value=[255, 255, 255])

    def draw_contours(self):
        self.image_with_contours = self.image.copy()
        cv2.drawContours(self.image_with_contours, [self.contour_with_max_area], -1, (0, 255, 0), 1)

    def calculateDistanceBetween2Points(self, p1, p2):
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis

    def order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def store_process_image(self, file_name, image):
        path = "./process_images/table_extractor/" + file_name
        cv2.imwrite(path, image)

    def crop(self):
        y = self.perspective_corrected_image.shape[0]
        x = self.perspective_corrected_image.shape[1]
        x2 = int(x - x * 0.04)
        x1 = int(0 + x * 0.04)
        self.cropped_image = self.perspective_corrected_image[:, x1:x2]


if __name__ == '__main__':
    file_name = "img/01.jpg"
    TE = TableExtractor(file_name)
    perspective_corrected_image_with_padding, contour_with_max_area = TE.execute("all_tables")

