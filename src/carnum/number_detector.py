import cv2
from cv2.typing import MatLike

from src.carnum import BoundingBox
from src.carnum import NumberCandidate


class NumberDetector:
    def __init__(
        self,
        img: MatLike,
        contrast_clip_limit: float = 3,
        contrast_kernel_size: int = 8,
        target_img_width: int = 1920,
        target_img_height: int = 1080,
        dilation_kernel_size: int = 3,
    ):
        self.img: MatLike = img
        self.edges: MatLike

        self.contrast_clip_limit: float = contrast_clip_limit
        self.contrast_kernel_size: int = contrast_kernel_size
        self.target_img_width: int = target_img_width
        self.target_img_height: int = target_img_height
        self.dilation_kernel_size: int = dilation_kernel_size

    def detect_number(self) -> NumberCandidate | None:
        self.__preprocess_img()
        candidates = self.__find_number_candidates()
        return self.__select_best_candidate(candidates)

    def __preprocess_img(self) -> None:
        self.__enhance_contrast()
        self.img = cv2.bilateralFilter(self.img, 3, 25, 75)
        self.img, _ = self.resize_to_target(self.img)

    def __find_number_candidates(self) -> list[NumberCandidate]:
        self.edges = cv2.Canny(self.img, 100, 200)
        self.__morphology_dilation()
        return self.__find_contours()

    def __enhance_contrast(self) -> None:
        """
        Улучшение контрастности
        """
        # CLAHE для улучшения локального контраста
        clahe = cv2.createCLAHE(self.contrast_clip_limit, (self.contrast_kernel_size, self.contrast_kernel_size))
        self.img = clahe.apply(self.img)

    def resize_to_target(self, img: MatLike) -> tuple[MatLike, float]:
        """
        Приведение изображения к целевому размеру с сохранением пропорций
        """
        height, width = img.shape[:2]

        print(f'Исходный размер: {width}x{height}')

        scale_x = self.target_img_width / width
        scale_y = self.target_img_height / height

        scale = min(scale_x, scale_y)

        if scale <= 1:
            print('Изображение уже больше целевого размера, оставляем как есть')
            return img, 1.0

        new_width = int(width * scale)
        new_height = int(height * scale)

        print(f'Новый размер: {new_width}x{new_height}, масштаб: {scale:.2f}')

        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        return img, scale

    def __morphology_dilation(self) -> None:
        """
         Утолщение границ с помощью морфологической дилатации
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dilation_kernel_size, self.dilation_kernel_size))
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
                BoundingBox(x, y, w, h),
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
