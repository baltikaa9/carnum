from dataclasses import dataclass
from typing import override

from cv2.typing import MatLike

from src.carnum import BoundingBox


@dataclass
class NumberCandidate:
    contour: MatLike
    bbox: BoundingBox
    area: float
    aspect_ratio: float

    @override
    def __str__(self):
        return f'{self.bbox.x=} {self.bbox.y=} {self.bbox.w}x{self.bbox.h}, соотношение: {self.aspect_ratio:.2f}'
