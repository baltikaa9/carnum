from dataclasses import dataclass

import numpy as np

from . import BorderBox


@dataclass
class NumberCandidate:
    contour: np.ndarray
    bbox: BorderBox
    area: float
    aspect_ratio: float

    def __str__(self):
        return f'{self.bbox.x=} {self.bbox.y=} {self.bbox.w}x{self.bbox.h}, соотношение: {self.aspect_ratio:.2f}'