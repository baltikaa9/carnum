import cv2
from cv2.typing import MatLike
import numpy as np

def generate_digit_template(digit: str, size=(48, 32)) -> MatLike:
    img = np.ones(size, dtype=np.uint8) * 255
    # Используем шрифт, близкий к номерам — например, "Arial" или "Digital"
    cv2.putText(img, digit, (-2, 42), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 0), 2)
    # _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img


if __name__ == '__main__':
    for d in '0123456789':
        tmpl = generate_digit_template(d)
        r = cv2.imwrite(f'img/templates/{d}.png', tmpl)
        print(r)
