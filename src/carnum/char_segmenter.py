from collections.abc import Sequence
import cv2
from cv2.typing import MatLike

from src.carnum.border_box import BoundingBox

class CharSegmenter:
    def __init__(self, number_img: MatLike):
        self.img: MatLike = number_img

    def segment_characters(self) -> list[MatLike]:
        img = self.__preprocess(self.img)

        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = self.__filter_contours(contours)

        boxes = sorted(boxes, key=lambda b: b.x)

        return self.__crop_characters(img, boxes)

    def __preprocess(self, img: MatLike) -> MatLike:
        binary = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=13,
            C=3,
        )

        # Удаляем мелкие шумы (точки)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        return binary

    def __filter_contours(self, contours: Sequence[MatLike]) -> list[BoundingBox]:
        h_img, w_img = self.img.shape
        boxes: list[BoundingBox] = []
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

            boxes.append(BoundingBox(x, y, w, h))

        return boxes

    def __crop_characters(self, img: MatLike, boxes: list[BoundingBox]) -> list[MatLike]:
        chars: list[MatLike] = []
        for box in boxes:
            char_img = img[box.y:box.y+box.h, box.x:box.x+box.w]
            chars.append(char_img)
        return chars
