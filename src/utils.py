import os

# Глобальные счётчики для совпавших, несовпавших углов и случаев, когда сравнение не производится
total_matched = 0
total_unmatched = 0
total_not_compared = 0


def compare_angle(func):
    """
    Декоратор, который после выполнения метода сравнивает вычисленный угол self.median_filtered_angle
    с ожидаемым значением, взятым из соответствующего .txt файла в папке 'angles'.
    Также обновляет глобальные счётчики:
      - total_matched: когда разница меньше 1 градуса,
      - total_unmatched: когда разница 1 градус и более,
      - total_not_compared: когда файл с ожидаемым углом не найден или медианный угол не вычислен.
    """
    def wrapper(self, *args, **kwargs):
        global total_matched, total_unmatched, total_not_compared
        result = func(self, *args, **kwargs)
        # Проверяем, вычислен ли медианный угол
        if self.median_angle is not None:
            # Формируем имя файла с ожидаемым углом (заменяем расширение на .txt)
            base_name, _ = os.path.splitext(self.file_name)
            expected_angle_path = os.path.join("angles", base_name + ".txt")

            # Приводим вычисленный угол к единому формату
            if self.median_angle > 45:
                corrected = self.median_angle - 90
            elif self.median_angle < -45:
                corrected = self.median_angle + 90
            else:
                corrected = self.median_angle
            corrected = -corrected

            if os.path.exists(expected_angle_path):
                try:
                    with open(expected_angle_path, "r") as f:
                        expected_angle = float(f.read().strip())
                    difference = abs(corrected - expected_angle)
                    print(f"Сравнение углов для {self.file_name}: вычисленный угол = {corrected}, "
                          f"ожидаемый угол = {expected_angle}, разница = {difference}")
                    # Обновляем счётчики и выводим сообщение
                    if difference < 1:
                        print("Угол совпал")
                        total_matched += 1
                    else:
                        print("Угол не совпал")
                        total_unmatched += 1
                except Exception as e:
                    print(f"Ошибка при чтении или обработке файла {expected_angle_path}: {e}")
                    total_not_compared += 1
            else:
                print(f"Файл с ожидаемым углом не найден: {expected_angle_path}")
                total_not_compared += 1
        else:
            print("Медианный угол не вычислен, сравнение не производится.")
            total_not_compared += 1
        return result
    return wrapper