import os

import cv2
from cv2.typing import MatLike
import matplotlib
import matplotlib.pyplot as plt
from pytesseract import image_to_string

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

def recognize_letter(symbol_img: MatLike) -> str:
    """
    Распознаёт один символ с помощью Tesseract.
    """
    # Убираем шум по краям (опционально)
    # symbol_img = cv2.copyMakeBorder(symbol_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)

    # Распознаём как один символ
    char = image_to_string(
        symbol_img,
        lang='eng',
        config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABEKMHOPCTYX0123456789'
    ).strip()

    # Tesseract иногда возвращает "1" вместо "А" и т.п. — можно добавить пост-обработку
    return char

def fix_digits(char: str, is_digit_position: bool) -> str:
    if is_digit_position:
        # На позиции цифры: исправляем буквы → цифры
        fixes = {
            'O': '0',
            'E': '3',
            'A': '4',
            'T': '7',
            'B': '8',
            'P': '9',
            'K': '3',  # частая ошибка: K → 3
        }
        return fixes.get(char, char)
    else:
        # На позиции буквы: исправляем цифры → буквы
        fixes = {
            '0': 'O',
            '3': 'E',
            '4': 'У',
            '6': 'B',  # иногда 6 → B
            '7': 'T',
            '8': 'B',
        }
        return fixes.get(char, char)

def resize_with_padding(img, target_size=(32, 48)):
    h, w = img.shape
    target_w, target_h = target_size

    # Сохраняем пропорции
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Добавляем белый padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
    return padded

def recognize_digit(symbol_img: MatLike, templates: dict[str, MatLike]) -> str:
    """
    Распознаёт цифру методом сравнения с шаблонами.
    """
    # Приведи к одному размеру
    # resized = resize_with_padding(symbol_img, (32, 48))
    resized = cv2.resize(symbol_img, (32, 48))
    _, resized = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY)
    resized = cv2.medianBlur(resized, 3)

    # kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    # resized = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel_open)

    fig = plt.figure()
    ax = fig.add_subplot(341)
    ax.imshow(resized, cmap='gray')

    # print(symbol_img.shape)
    # print(templates['0'].shape)

    best_match = None
    best_score = -1

    for digit, tmpl in templates.items():
        ax = fig.add_subplot(3, 4, int(digit) + 2)
        ax.imshow(tmpl, cmap='gray')
        # Сравниваем по корреляции (чем ближе к 1 — тем лучше)
        match = cv2.matchTemplate(symbol_img, tmpl, cv2.TM_CCOEFF_NORMED)
        score = match[0][0]
        print(f'{score} {digit}')
        if score > best_score:
            best_score = score
            best_match = digit

    # Порог: если совпадение слабое — не доверяем
    if best_score < 0.4:
        return '?'  # неизвестно
    return best_match

def load_templates() -> dict[str, MatLike]:
    templates: dict[str, MatLike] = {}
    for d in '0123456789':
        path = f'img/templates/{d}.png'
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates[d] = img
    return templates

def recognize_number(symbols: list[MatLike], templates: dict[str, MatLike]) -> str:
    raw_chars = []
    # for sym in symbols:
    for i, c in enumerate(symbols):
        # is_digit_pos = i not in [0, 4, 5]  # кроме 0,4,5 — цифры
        # is_digit_pos = i == 1
        char = recognize_letter(c)
        # if not is_digit_pos:
            # char = recognize_letter(c)
        # else:
            # char = recognize_digit(c, templates)

        raw_chars.append(char)

    text = ''.join(raw_chars)
    print('Сырой результат:', text)

    # Исправление по позициям (пример для 8-символьного номера)
    # corrected = ''
    # for i, c in enumerate(text):
    #     is_digit_pos = i not in [0, 4, 5]  # кроме 0,4,5 — цифры

    #     corrected += fix_digits(c, is_digit_pos)

    return text

def main():
    # TODO: Сегментация символов
    # TODO: Распознавание символов
    # img = cv2.imread('img/01-715.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('img/154yn1QYKvMGFzWM75SG8NjK64po-CwRLOsLqI4-4sI8yNuiOS1qpod1d_8sk8YFsygRv5QLsLgnc1uJhskSEg1.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('img/14.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('img/2025-01-16 23.01.30.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('img/fine1.jpg', cv2.IMREAD_GRAYSCALE)

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

    templates = load_templates()

    print(recognize_number(chars, templates))

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
