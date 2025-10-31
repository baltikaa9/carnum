from PySide6.QtWidgets import QMainWindow, QVBoxLayout
from cv2.typing import MatLike
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from .ui.ui_main_window import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Загружаем UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setup_plot()

    def setup_plot(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self.ui.graphic_widget)
        layout.addWidget(self.canvas)

    def imshow(self, edges: MatLike, contour_img: MatLike, number_img: MatLike, chars: list[MatLike]):
        self.figure.clear()
        ax1 = self.figure.add_subplot(231)
        ax1.set_xticks([]), ax1.set_yticks([])
        ax2 = self.figure.add_subplot(232)
        ax2.set_xticks([]), ax2.set_yticks([])
        ax3 = self.figure.add_subplot(233)
        ax3.set_xticks([]), ax3.set_yticks([])

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
