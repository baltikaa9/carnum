import os

import cv2
from cv2.typing import MatLike
import matplotlib
import matplotlib.pyplot as plt

from src.carnum import BorderBox
from src.carnum import NumberDetector

matplotlib.use('GTK4Agg')
os.environ['QT_QPA_PLATFORM'] = 'xcb'

def draw_contour_and_bbox(img: MatLike, contour: MatLike, bbox: BorderBox) -> tuple[MatLike, MatLike]:
    """
    Рисуем контуры и bounding boxes на изображении
    """
    # Создаем копии для визуализации
    contour_img = img.copy()
    bbox_img = img.copy()

    # Рисуем все найденные контуры
    cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 3)

    cv2.rectangle(bbox_img, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), (0, 255, 0), 3)

    return contour_img, bbox_img

def segment_characters(number_img: MatLike) -> list[MatLike]:
    binary = cv2.adaptiveThreshold(
        number_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=13,
        C=3,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks([]), plt.yticks([])

    # 2. Морфологическая очистка
    # Удаляем мелкие шумы (точки)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # Закрываем мелкие разрывы внутри символов (например, в "8", "Б")
    # kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    ax.imshow(binary, cmap='gray')
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Фильтрация контуров
    h_img, w_img = binary.shape
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 5 or h < 10:
            continue

        if h <= 0.35 * h_img or h >= 0.8 * h_img:
            continue

        if w > 0.5 * w_img:  # один символ не может занимать больше половины ширины номера
            continue

        aspect_ratio = w / float(h)
        if aspect_ratio > 1.2:
            continue

        boxes.append((x, y, w, h))

    # 5. Сортируем слева направо
    boxes = sorted(boxes, key=lambda b: b[0])

    chars = []
    for (x, y, w, h) in boxes:
        char_img = binary[y:y+h, x:x+w]
        chars.append(char_img)

    return chars

def main():
    # TODO: Сегментация символов
    # TODO: Распознавание символов
    # img = cv2.imread('img/01-715.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('img/154yn1QYKvMGFzWM75SG8NjK64po-CwRLOsLqI4-4sI8yNuiOS1qpod1d_8sk8YFsygRv5QLsLgnc1uJhskSEg1.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('img/14.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('img/PVI_LTtE9Zu3BtFoud-W58xsg2MN3kAfNZA0GZwR0qNdTAhdbDdRwVYHic9fcY5yayS5PezuRW74LI-RFeIxCw.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('img/fine1.jpg', cv2.IMREAD_GRAYSCALE)

    assert img is not None, 'file could not be read, check with os.path.exists()'

    detector = NumberDetector(img)
    number_candidate = detector.detect_number()

    assert number_candidate is not None, 'Не удалось распознать номер'

    x, y, w, h = number_candidate.bbox
    number_img = detector.img[y:y + h, x:x + w]

    # number_img_res, _ = detector.resize_to_target(number_img)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.xticks([]), plt.yticks([])
    # ax.imshow(binary, cmap='gray')

    chars = segment_characters(number_img)
    print(len(chars))

    contour_img, bbox_img = draw_contour_and_bbox(detector.img, number_candidate.contour, number_candidate.bbox)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 4, 1)
    # plt.xticks([]), plt.yticks([])
    # ax2 = fig.add_subplot(1, 4, 2)
    # plt.xticks([]), plt.yticks([])
    # ax3 = fig.add_subplot(1, 4, 3)
    # plt.xticks([]), plt.yticks([])
    # ax4 = fig.add_subplot(1, 4, 4)
    # plt.xticks([]), plt.yticks([])

    # ax1.imshow(detector.edges, cmap='gray')
    # ax2.imshow(contour_img, cmap='gray')
    # ax3.imshow(bbox_img, cmap='gray')
    # ax4.imshow(number_img, cmap='gray')

    # ax = fig.add_subplot(111)
    # plt.xticks([]), plt.yticks([])
    # ax.imshow(binary, cmap='gray')

    n = len(chars)

    fig, axes = plt.subplots(1, n, figsize=(n * 1.2, 2))
    if n == 1:
        axes = [axes]
    for ax, sym in zip(axes, chars):
        ax.imshow(sym, cmap='gray')
        ax.axis('off')
    plt.suptitle(f'Найдено {n} символов')

    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()
