# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.10.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QPushButton, QSizePolicy,
    QSpacerItem, QToolButton, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_3 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.graphic_widget = QWidget(self.centralwidget)
        self.graphic_widget.setObjectName(u"graphic_widget")

        self.horizontalLayout_3.addWidget(self.graphic_widget)

        self.control_widget = QWidget(self.centralwidget)
        self.control_widget.setObjectName(u"control_widget")
        self.control_widget.setMaximumSize(QSize(300, 16777215))
        self.verticalLayout = QVBoxLayout(self.control_widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupBox = QGroupBox(self.control_widget)
        self.groupBox.setObjectName(u"groupBox")
        self.horizontalLayout_4 = QHBoxLayout(self.groupBox)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.input_path = QLineEdit(self.groupBox)
        self.input_path.setObjectName(u"input_path")

        self.horizontalLayout_4.addWidget(self.input_path)

        self.select_path = QToolButton(self.groupBox)
        self.select_path.setObjectName(u"select_path")

        self.horizontalLayout_4.addWidget(self.select_path)


        self.verticalLayout.addWidget(self.groupBox)

        self.widget = QWidget(self.control_widget)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout_5 = QHBoxLayout(self.widget)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")

        self.horizontalLayout_5.addWidget(self.label)

        self.output_number = QLineEdit(self.widget)
        self.output_number.setObjectName(u"output_number")
        self.output_number.setEnabled(False)

        self.horizontalLayout_5.addWidget(self.output_number)


        self.verticalLayout.addWidget(self.widget)

        self.pushButton = QPushButton(self.control_widget)
        self.pushButton.setObjectName(u"pushButton")

        self.verticalLayout.addWidget(self.pushButton)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.horizontalLayout_3.addWidget(self.control_widget)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0441\u043f\u043e\u043b\u043e\u0436\u0435\u043d\u0438\u0435 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f", None))
        self.select_path.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0441\u043f\u043e\u0437\u043d\u0430\u043d\u043d\u044b\u0439 \u043d\u043e\u043c\u0435\u0440", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0441\u043f\u043e\u0437\u043d\u0430\u0442\u044c", None))
    # retranslateUi

