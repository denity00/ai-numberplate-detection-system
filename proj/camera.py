import cv2
import time

# Открываем камеру
cap = cv2.VideoCapture(0)  # 0 - это индекс камеры, если у вас одна камера

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру.")
    exit()

try:
    while True:
        # Захватываем кадр
        ret, frame = cap.read()

        if not ret:
            print("Ошибка: Не удалось захватить кадр.")
            break

        # Сохраняем кадр в файл (или обрабатываем его)
        timestamp = int(time.time())
        filename = f"frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Сохранен кадр: {filename}")

        # Ждем 10 секунд
        time.sleep(10)

except KeyboardInterrupt:
    print("Скрипт остановлен пользователем.")

finally:
    # Освобождаем камеру
    cap.release()
    print("Камера освобождена.")