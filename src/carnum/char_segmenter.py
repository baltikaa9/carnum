import cv2
from cv2.typing import MatLike
import numpy as np
import matplotlib.pyplot as plt

class CharSegmenter:
    def __init__(self, img: MatLike):
        self.img: MatLike = img

    def __preprocess(self):
        # Convert to grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        return thresh
