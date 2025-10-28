import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os

matplotlib.use('GTK4Agg')

def sobel(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

    return cv2.filter2D(img, -1, kernel)

def canny(img: np.ndarray) -> np.ndarray:
    img_blur_5 = cv2.GaussianBlur(img, (5, 5), 0)
    return sobel(img_blur_5)

def main():
    img = cv2.imread('img/14.jpg', cv2.IMREAD_GRAYSCALE)

    assert img is not None, 'file could not be read, check with os.path.exists()'

    edges_canny = cv2.Canny(img, 100, 200)

    edges_canny_2 = canny(img)

    fig = plt.figure()
    # ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2)

    # ax1.imshow(img, cmap='gray')
    ax2.imshow(edges_canny, cmap='gray')
    ax3.imshow(edges_canny_2, cmap='gray')

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()