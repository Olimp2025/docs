import os


def compare_angle(func):
    """
    Декоратор, который после выполнения метода сравнивает вычисленный угол self.median_filtered_angle
    с ожидаемым значением, взятым из соответствующего .txt файла в папке 'rotation_angles'.
    """
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        # После выполнения метода проверяем, вычислен ли угол
        if self.median_filtered_angle is not None:
            # Формируем имя файла с ожидаемым углом: заменяем расширение на .txt
            base_name, _ = os.path.splitext(self.file_name)
            expected_angle_path = os.path.join("rotation_angles", base_name + ".txt")

            # Привожу углы к одному формату
            if self.median_filtered_angle > 45:
                corrected = self.median_filtered_angle - 90
            elif self.median_filtered_angle < -45:
                corrected = self.median_filtered_angle + 90
            else:
                corrected = self.median_filtered_angle
            corrected = -corrected

            if os.path.exists(expected_angle_path):
                try:
                    with open(expected_angle_path, "r") as f:
                        expected_angle = float(f.read().strip())
                    difference = abs(corrected - expected_angle)
                    print(f"Сравнение углов для {self.file_name}: вычисленный угол = {corrected}, "
                          f"ожидаемый угол = {expected_angle}, разница = {difference}")
                except Exception as e:
                    print(f"Ошибка при чтении или обработке файла {expected_angle_path}: {e}")
            else:
                print(f"Файл с ожидаемым углом не найден: {expected_angle_path}")
        else:
            print("Медианный угол не вычислен, сравнение не производится.")
        return result
    return wrapper


