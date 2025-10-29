import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.carnum import BorderBox
from src.carnum import NumberDetector

matplotlib.use('GTK4Agg')


def draw_contour_and_bbox(img: np.ndarray, contour: np.ndarray, bbox: BorderBox) -> tuple[np.ndarray, np.ndarray]:
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

    number_img = detector.detect_number()

    assert number_img is not None, 'Не удалось распознать номер'

    x, y, w, h = number_img.bbox

    contour_img, bbox_img = draw_contour_and_bbox(detector.img, number_img.contour, number_img.bbox)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 4, 1)
    plt.xticks([]), plt.yticks([])
    ax2 = fig.add_subplot(1, 4, 2)
    plt.xticks([]), plt.yticks([])
    ax3 = fig.add_subplot(1, 4, 3)
    plt.xticks([]), plt.yticks([])
    ax4 = fig.add_subplot(1, 4, 4)
    plt.xticks([]), plt.yticks([])

    ax1.imshow(detector.edges, cmap='gray')
    ax2.imshow(contour_img, cmap='gray')
    ax3.imshow(bbox_img, cmap='gray')
    ax4.imshow(detector.img[y:y + h, x:x + w], cmap='gray')

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
