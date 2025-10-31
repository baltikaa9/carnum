import os
import sys

from PySide6.QtWidgets import QApplication
import cv2
from cv2.typing import MatLike
import matplotlib
import matplotlib.pyplot as plt

from src.carnum.main_window import MainWindow
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
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # return
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

    chars = segmenter.segment_characters()

    recognizer = CharRecognizer(chars, load_templates())

    print(recognizer.recognize())

    contour_img, _ = draw_contour_and_bbox(detector.img, number_candidate.contour, number_candidate.bbox)

    window.imshow(detector.edges, contour_img, number_img, chars)

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
