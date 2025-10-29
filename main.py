import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os

matplotlib.use('GTK4Agg')


def sobel(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    return cv2.filter2D(img, -1, kernel)


def find_contours(edges: np.ndarray) -> list[dict]:
    """
    Поиск контуров-кандидатов в номерные пластины
    """
    # Поиск всех контуров
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,
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
        if area < 1000:  # Минимальная площадь для номерного знака
            continue

        # Аппроксимируем контур до более простой формы
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * perimeter  # Точность аппроксимации
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # print(i, approx)

        # Нас интересуют контуры с 4 углами (прямоугольники)
        # if 4 <= len(approx) <= 6:
            # Получаем bounding box
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

            # Фильтруем по пропорциям (типичные для номерных знаков)
            # if 3.5 <= aspect_ratio <= 5:
        candidates.append({
            'contour': approx,
            'bbox': (x, y, w, h),
            'area': area,
            'aspect_ratio': aspect_ratio
        })
        print(f"Кандидат {len(candidates)}: {x=} {y=} {w}x{h}, соотношение: {aspect_ratio:.2f}")

    return candidates


def select_best_candidate(candidates: list[dict], img: np.ndarray) -> dict:
    """
    Продвинутый выбор кандидата с анализом текстуры
    """
    if not candidates:
        return {}

    img_height, img_width = img.shape
    best_candidate = None
    best_score = -1

    for candidate in candidates:
        score = 0
        x, y, w, h = candidate['bbox']

        # Базовые критерии
        aspect_ratio = candidate['aspect_ratio']
        area = candidate['area']
        img_area = img_width * img_height
        area_ratio = area / img_area

        # 1. Соотношение сторон (самый важный критерий)
        if 4.0 <= aspect_ratio <= 5.0:
            score += 4
        elif 3.5 <= aspect_ratio <= 5.5:
            score += 2
        elif 2.5 <= aspect_ratio <= 6.0:
            score += 1

        # 2. Площадь
        if 0.005 <= area_ratio <= 0.05:
            score += 3
        elif 0.001 <= area_ratio <= 0.1:
            score += 1

        # 3. Положение
        center_y = y + h / 2
        if center_y > img_height * 0.4:
            score += 2

        # 4. Анализ текстуры региона
        # texture_score = analyze_candidate_region(img, candidate)
        # if texture_score > 30:  # Эмпирический порог
        #     score += 3
        # elif texture_score > 20:
        #     score += 1

        # 5. Форма контура
        if 4 <= len(candidate['contour']) <= 6:
            score += 2

        # 6. Размер (абсолютные значения)
        # if w > 100 and h > 20:  # Минимальные размеры
        #     score += 1

        candidate['score'] = score
        # candidate['texture_score'] = texture_score

        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate:
        x, y, w, h = best_candidate['bbox']
        print(
            f"Лучший кандидат: {x=} {y=} {w}x{h}, "
            f"score: {best_score}"
        )

    return best_candidate


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

def draw_contour_and_bbox(original_img: np.ndarray, contour: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Рисуем контуры и bounding boxes на изображении
    """
    # Создаем копии для визуализации
    contour_img = original_img.copy()
    bbox_img = original_img.copy()

    # Рисуем все найденные контуры
    cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 3)

    x, y, w, h = bbox
    cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return contour_img, bbox_img

def enhance_contrast(img: np.ndarray, clip_limit: float = 2, kernel_size: int = 8) -> np.ndarray:
    """
    Улучшение контрастности специально для номерных знаков
    """
    # CLAHE для улучшения локального контраста
    clahe = cv2.createCLAHE(clip_limit, (kernel_size, kernel_size))
    return clahe.apply(img)


def morphology_dilation(edges: np.ndarray, kernel_size=2) -> np.ndarray:
    """
     Утолщение границ с помощью морфологической дилатации
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(edges, kernel, iterations=1)

def upscale(img: np.ndarray, scale_factor=2) -> np.ndarray:
    height, width = img.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def resize_to_target(image, target_width=1920, target_height=1080):
    """
    Приведение изображения к целевому размеру с сохранением пропорций
    """
    # Получаем текущие размеры
    height, width = image.shape[:2]

    print(f"Исходный размер: {width}x{height}")

    # Вычисляем коэффициенты масштабирования
    scale_x = target_width / width
    scale_y = target_height / height

    # Выбираем минимальный коэффициент, чтобы изображение полностью поместилось
    scale = min(scale_x, scale_y)

    # Если изображение уже меньше целевого размера - не уменьшаем его
    if scale <= 1:
        print("Изображение уже больше целевого размера, оставляем как есть")
        return image, 1.0

    # Вычисляем новые размеры
    new_width = int(width * scale)
    new_height = int(height * scale)

    print(f"Новый размер: {new_width}x{new_height}, масштаб: {scale:.2f}")

    # Масштабируем изображение
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return resized, scale

def main():
    # img = cv2.imread('img/01-715.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('img/154yn1QYKvMGFzWM75SG8NjK64po-CwRLOsLqI4-4sI8yNuiOS1qpod1d_8sk8YFsygRv5QLsLgnc1uJhskSEg1.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('img/14.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('img/PVI_LTtE9Zu3BtFoud-W58xsg2MN3kAfNZA0GZwR0qNdTAhdbDdRwVYHic9fcY5yayS5PezuRW74LI-RFeIxCw.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('img/fine1.jpg', cv2.IMREAD_GRAYSCALE)

    assert img is not None, 'file could not be read, check with os.path.exists()'

    img = enhance_contrast(img, 3)

    # img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.bilateralFilter(img, 3, 25, 75)

    # img = upscale(img, 3)
    img, scale = resize_to_target(img)

    edges = cv2.Canny(img, 100, 200)

    edges = morphology_dilation(edges, 3)

    candidates = find_contours(edges)

    best_candidate = select_best_candidate(candidates, img)

    x, y, w, h = best_candidate['bbox']

    # contour_img, bbox_img = draw_contours_and_bboxes(img, [best_candidate])
    contour_img, bbox_img = draw_contour_and_bbox(img, best_candidate['contour'], best_candidate['bbox'])
    # print(candidates)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 4, 1)
    plt.xticks([]), plt.yticks([])
    ax2 = fig.add_subplot(1, 4, 2)
    plt.xticks([]), plt.yticks([])
    ax3 = fig.add_subplot(1, 4, 3)
    plt.xticks([]), plt.yticks([])
    ax4 = fig.add_subplot(1, 4, 4)
    plt.xticks([]), plt.yticks([])

    ax1.imshow(edges, cmap='gray')
    ax2.imshow(contour_img, cmap='gray')
    ax3.imshow(bbox_img, cmap='gray')
    ax4.imshow(img[y:y + h, x:x + w], cmap='gray')

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
