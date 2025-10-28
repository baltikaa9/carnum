import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os

matplotlib.use('GTK4Agg')


def sobel(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    return cv2.filter2D(img, -1, kernel)


def canny(img: np.ndarray) -> np.ndarray:
    img_blur_5 = cv2.GaussianBlur(img, (5, 5), 0)
    return sobel(img_blur_5)


def find_license_plate_contours(edges: np.ndarray) -> list[dict]:
    """
    Поиск контуров-кандидатов в номерные пластины
    """
    # Поиск всех контуров
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    print(f"Найдено контуров: {len(contours)}")

    # print(contours)

    # Сортируем контуры по площади (от больших к маленьким)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    candidates = []

    # Анализируем каждый контур
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # Пропускаем слишком маленькие контуры
        if area < 100:  # Минимальная площадь для номерного знака
            continue

        # Аппроксимируем контур до более простой формы
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * perimeter  # Точность аппроксимации
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # print(i, approx)

        # Нас интересуют контуры с 4 углами (прямоугольники)
        if len(approx) == 4:
            # Получаем bounding box
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Фильтруем по пропорциям (типичные для номерных знаков)
            if 2 <= aspect_ratio <= 5:
                candidates.append({
                    'contour': approx,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })
                print(f"Кандидат {len(candidates)}: {x=} {y=} {w}x{h}, соотношение: {aspect_ratio:.2f}")

    return candidates


def draw_contours_and_bboxes(original_img: np.ndarray, candidates: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Рисуем контуры и bounding boxes на изображении
    """
    # Создаем копии для визуализации
    contour_img = original_img.copy()
    bbox_img = original_img.copy()

    # Рисуем все найденные контуры
    cv2.drawContours(contour_img, [c['contour'] for c in candidates], -1, (0, 255, 0), 3)

    # Рисуем bounding boxes с номерами
    for i, candidate in enumerate(candidates):
        x, y, w, h = candidate['bbox']
        # Рисуем прямоугольник
        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Добавляем номер кандидата
        cv2.putText(bbox_img, f'{i + 1}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return contour_img, bbox_img


def adaptive_canny(img: np.ndarray, sigma=0.33) -> np.ndarray:
    """
    Адаптивный Canny с автоматическим подбором порогов
    """
    # Вычисляем медиану интенсивности изображения
    v = np.median(img)

    # Автоматически определяем пороги на основе медианы
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    print(f"Адаптивные пороги Canny: lower={lower}, upper={upper}")

    # Применяем Canny
    return cv2.Canny(img, lower, upper)

def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    Улучшение контрастности специально для номерных знаков
    """
    # CLAHE для улучшения локального контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def smart_morphology(edges: np.ndarray) -> np.ndarray:
    """
    Умные морфологические операции для соединения обрывов
    """
    # Сначала закрытие для соединения близких границ
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    return closed

    # Затем дилатация для утолщения линий
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(closed, kernel_dilate, iterations=1)

def main():
    # img = cv2.imread('img/01-715.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('img/154yn1QYKvMGFzWM75SG8NjK64po-CwRLOsLqI4-4sI8yNuiOS1qpod1d_8sk8YFsygRv5QLsLgnc1uJhskSEg.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('img/14.jpg', cv2.IMREAD_GRAYSCALE)

    assert img is not None, 'file could not be read, check with os.path.exists()'

    img = enhance_contrast(img)

    edges_canny = cv2.Canny(img, 100, 200)
    # edges_canny_2 = canny(img)
    # edges_canny = adaptive_canny(img)
    # edges = smart_morphology(edges_canny)
    edges = edges_canny

    candidates = find_license_plate_contours(edges)
    contour_img, bbox_img = draw_contours_and_bboxes(img, candidates)
    # print(candidates)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    plt.xticks([]), plt.yticks([])
    ax2 = fig.add_subplot(1, 3, 2)
    plt.xticks([]), plt.yticks([])
    ax3 = fig.add_subplot(1, 3, 3)
    plt.xticks([]), plt.yticks([])

    ax1.imshow(edges, cmap='gray')
    ax2.imshow(contour_img, cmap='gray')
    ax3.imshow(bbox_img, cmap='gray')

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
