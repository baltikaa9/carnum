from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QVBoxLayout
import cv2
from cv2.typing import MatLike
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from src.carnum import BoundingBox, NumberDetector
from src.carnum import CharRecognizer
from src.carnum import CharSegmenter

from .ui.ui_main_window import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.select_path.clicked.connect(self.select_path)
        self.ui.pushButton.clicked.connect(self.start)

        self.setup_plot()

    def setup_plot(self):
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self.ui.graphic_widget)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)

    def imshow(self, edges: MatLike, contour_img: MatLike, number_img: MatLike, chars: list[MatLike]):
        self.figure.clear()
        ax1 = self.figure.add_subplot(231)
        ax1.set_xticks([]), ax1.set_yticks([])
        ax1.set_title('Выделенные границы')
        ax2 = self.figure.add_subplot(232)
        ax2.set_xticks([]), ax2.set_yticks([])
        ax2.set_title('Найденный контур номера')
        ax3 = self.figure.add_subplot(233)
        ax3.set_xticks([]), ax3.set_yticks([])
        ax3.set_title('Вырезанный номер')

        ax1.imshow(edges, cmap='gray')
        ax2.imshow(contour_img, cmap='gray')
        ax3.imshow(number_img, cmap='gray')

        n = len(chars)
        if len(chars) == 0:
            print('No characters found')
            return

        for i in range(1, n + 1):
            ax = self.figure.add_subplot(2, n, n + i)
            ax.imshow(chars[i-1], cmap='gray')

        self.canvas.draw()

    def select_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Выберите изображение с номером',
            '',  # начальная директория (можно '~/Pictures' или '')
            'Изображения (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;Все файлы (*)'
        )
        if file_path:
            self.ui.input_path.setText(file_path)

    def load_templates(self) -> dict[str, MatLike]:
        templates: dict[str, MatLike] = {}
        for d in '0123456789':
            path = f'img/templates/{d}.png'
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates[d] = img
        return templates

    def draw_contour_and_bbox(self, img: MatLike, contour: MatLike, bbox: BoundingBox) -> tuple[MatLike, MatLike]:
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

    def start(self):
        file_path = self.ui.input_path.text()
        if not file_path:
            QMessageBox.warning(self, 'Ошибка', 'Путь к изображению не указан')
            return

        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            assert img is not None, 'file could not be read, check with os.path.exists()'

            detector = NumberDetector(img)

            number_candidate = detector.detect_number()

            assert number_candidate is not None, 'Не удалось распознать номер'

            x, y, w, h = number_candidate.bbox
            number_img = detector.img[y:y + h, x:x + w]

            segmenter = CharSegmenter(number_img)

            chars = segmenter.segment_characters()

            recognizer = CharRecognizer(chars, self.load_templates())

            self.ui.output_number.setText(recognizer.recognize())

            contour_img, _ = self.draw_contour_and_bbox(detector.img, number_candidate.contour, number_candidate.bbox)

            self.imshow(detector.edges, contour_img, number_img, chars)
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', str(e))
