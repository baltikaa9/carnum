import cv2
from cv2.typing import MatLike
from pytesseract import image_to_string


class CharRecognizer:
    def __init__(self, symbols: list[MatLike], templates: dict[str, MatLike]) -> None:
        self.symbols: list[MatLike] = symbols
        self.templates: dict[str, MatLike] = templates

    def recognize(self) -> str:
        chars: list[str] = []
        for i, symbol in enumerate(self.symbols):
            if i in [0, 4, 5]:  # 0, 4, 5 - буквы
                char = self.__recognize_letter(symbol)
            else:
                char = self.__recognize_digit(symbol)

            chars.append(char)

        return ''.join(chars)

    def __recognize_letter(self, symbol_img: MatLike) -> str:
        """
        Распознаёт один символ с помощью Tesseract.
        """
        char = str(image_to_string(
            symbol_img,
            lang='eng',
            config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABEKMHOPCTYX0123456789'
        )).strip()

        # Tesseract иногда возвращает "1" вместо "А" и т.п. — можно добавить пост-обработку
        return self.__fix_letter(char)

    def __recognize_digit(self, symbol_img: MatLike) -> str:
        """
        Распознаёт цифру методом сравнения с шаблонами.
        """
        resized = cv2.resize(symbol_img, (32, 48))

        best_match, best_score = '?', -1

        for digit, tmpl in self.templates.items():
            match = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
            score: float = match[0][0]
            if score > best_score:
                best_score, best_match = score, digit

        return best_match

    def __fix_letter(self, char: str) -> str:
        match char:
            case '0': return 'O'
            case '4': return 'У'
            case '6': return 'B'
            case '7': return 'T'
            case '8': return 'B'
            case _: return char
