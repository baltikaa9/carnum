import os

import cv2
from cv2.typing import MatLike
import matplotlib
import matplotlib.pyplot as plt

from src.carnum.char_recognizer import CharRecognizer
from src.carnum.char_segmenter import CharSegmenter
from src.carnum import BoundingBox
from src.carnum import NumberDetector

matplotlib.use('GTK4Agg')
os.environ['QT_QPA_PLATFORM'] = 'xcb'

def draw_contour_and_bbox(img: MatLike, contour: MatLike, bbox: BoundingBox) -> tuple[MatLike, MatLike]:
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

def load_templates() -> dict[str, MatLike]:
    templates: dict[str, MatLike] = {}
    for d in '0123456789':
        path = f'img/templates/{d}.png'
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates[d] = img
    return templates

def main():
    # TODO: Сегментация символов
    # TODO: Распознавание символов
    # img = cv2.imread('img/01-715.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('img/154yn1QYKvMGFzWM75SG8NjK64po-CwRLOsLqI4-4sI8yNuiOS1qpod1d_8sk8YFsygRv5QLsLgnc1uJhskSEg1.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('img/14.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('img/2025-01-16 23.01.30.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('img/fine1.jpg', cv2.IMREAD_GRAYSCALE)

    assert img is not None, 'file could not be read, check with os.path.exists()'

    detector = NumberDetector(img)

    number_candidate = detector.detect_number()

    assert number_candidate is not None, 'Не удалось распознать номер'

    x, y, w, h = number_candidate.bbox
    number_img = detector.img[y:y + h, x:x + w]

    segmenter = CharSegmenter(number_img)
    # number_img_res, _ = detector.resize_to_target(number_img)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.xticks([]), plt.yticks([])
    # ax.imshow(binary, cmap='gray')

    chars = segmenter.segment_characters()

    recognizer = CharRecognizer(chars, load_templates())

    print(recognizer.recognize())

    contour_img, bbox_img = draw_contour_and_bbox(detector.img, number_candidate.contour, number_candidate.bbox)

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    plt.xticks([]), plt.yticks([])
    ax2 = fig.add_subplot(222)
    plt.xticks([]), plt.yticks([])
    ax3 = fig.add_subplot(223)
    plt.xticks([]), plt.yticks([])
    ax4 = fig.add_subplot(224)
    plt.xticks([]), plt.yticks([])

    ax1.imshow(detector.edges, cmap='gray')
    ax2.imshow(contour_img, cmap='gray')
    ax3.imshow(bbox_img, cmap='gray')
    ax4.imshow(number_img, cmap='gray')

    # ax = fig.add_subplot(111)
    # plt.xticks([]), plt.yticks([])
    # ax.imshow(binary, cmap='gray')

    n = len(chars)

    if len(chars) == 0:
        print("No characters found")
        plt.tight_layout()

        plt.show()
        exit()

    fig, axes = plt.subplots(1, n, figsize=(n * 1.2, 2))
    if n == 1:
        axes = [axes]
    for ax, sym in zip(axes, chars):
        ax.imshow(sym, cmap='gray')
        # ax.axis('off')
    plt.suptitle(f'Найдено {n} символов')

    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()
