import cv2
import numpy as np

from . import BorderBox
from . import NumberCandidate


class NumberDetector:
    def __init__(self, img: np.ndarray):
        self.img: np.ndarray = img
        self.edges: np.ndarray | None = None

    def detect_number(self) -> NumberCandidate | None:
        self.__enhance_contrast(3)
        self.img = cv2.bilateralFilter(self.img, 3, 25, 75)
        self.__resize_to_target()

        self.edges = cv2.Canny(self.img, 100, 200)
        self.__morphology_dilation(3)

        candidates = self.__find_contours()

        return self.__select_best_candidate(candidates)

    def __enhance_contrast(self, clip_limit: float = 2, kernel_size: int = 8) -> None:
        """
        Улучшение контрастности
        """
        # CLAHE для улучшения локального контраста
        clahe = cv2.createCLAHE(clip_limit, (kernel_size, kernel_size))
        self.img = clahe.apply(self.img)

    def __resize_to_target(self, target_width=1920, target_height=1080) -> float:
        """
        Приведение изображения к целевому размеру с сохранением пропорций
        """
        height, width = self.img.shape[:2]

        print(f'Исходный размер: {width}x{height}')

        scale_x = target_width / width
        scale_y = target_height / height

        scale = min(scale_x, scale_y)

        if scale <= 1:
            print('Изображение уже больше целевого размера, оставляем как есть')
            return 1.0

        new_width = int(width * scale)
        new_height = int(height * scale)

        print(f'Новый размер: {new_width}x{new_height}, масштаб: {scale:.2f}')

        self.img = cv2.resize(self.img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        return scale

    def __morphology_dilation(self, kernel_size=2) -> None:
        """
         Утолщение границ с помощью морфологической дилатации
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        self.edges = cv2.dilate(self.edges, kernel, iterations=1)

    def __find_contours(self) -> list[NumberCandidate]:
        """
        Поиск контуров-кандидатов в номерные пластины
        """
        contours, _ = cv2.findContours(
            self.edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,
        )

        print(f'Найдено контуров: {len(contours)}')

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        candidates: list[NumberCandidate] = []

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if area < 1000:
                continue

            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.02 * perimeter  # Точность аппроксимации
            approx = cv2.approxPolyDP(contour, epsilon, True)

            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            candidate = NumberCandidate(
                approx,
                BorderBox(x, y, w, h),
                area,
                aspect_ratio,
            )

            candidates.append(candidate)
            print(f'Кандидат {len(candidates)}: {candidate}')

        return candidates

    def __select_best_candidate(self, candidates: list[NumberCandidate]) -> NumberCandidate | None:
        if not candidates:
            return None

        img_height, img_width = self.img.shape
        best_candidate: NumberCandidate | None = None
        best_score = -1

        for candidate in candidates:
            score = 0
            bbox = candidate.bbox

            aspect_ratio = candidate.aspect_ratio
            area = candidate.area
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
            center_y = bbox.y + bbox.h / 2
            if center_y > img_height * 0.4:
                score += 2

            # 5. Форма контура
            if 4 <= len(candidate.contour) <= 6:
                score += 2

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate:
            print(f'Лучший кандидат: {best_candidate}, score: {best_score}')

        return best_candidate