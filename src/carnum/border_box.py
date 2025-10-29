from dataclasses import dataclass


@dataclass
class BorderBox:
    x: int
    y: int
    w: int
    h: int

    def __iter__(self):
        return iter([self.x, self.y, self.w, self.h])
