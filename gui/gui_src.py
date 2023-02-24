# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'gui_xml.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
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
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QSizePolicy,
    QTabWidget, QVBoxLayout, QWidget)

class Ui_gyarte(object):
    def setupUi(self, gyarte):
        if not gyarte.objectName():
            gyarte.setObjectName(u"gyarte")
        gyarte.resize(538, 432)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(gyarte.sizePolicy().hasHeightForWidth())
        gyarte.setSizePolicy(sizePolicy)
        gyarte.setMaximumSize(QSize(538, 432))
        gyarte.setTabShape(QTabWidget.Rounded)
        self.centralwidget = QWidget(gyarte)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.mode_selector_tab = QTabWidget(self.centralwidget)
        self.mode_selector_tab.setObjectName(u"mode_selector_tab")
        self.mode_selector_tab.setMinimumSize(QSize(0, 0))
        self.mode_selector_tab.setLayoutDirection(Qt.LeftToRight)
        self.mode_selector_tab.setStyleSheet(u"")
        self.mode_selector_tab.setTabPosition(QTabWidget.South)
        self.mode_selector_tab.setMovable(True)
        self.mode_selector_tab.setTabBarAutoHide(True)
        self.predict_tab = QWidget()
        self.predict_tab.setObjectName(u"predict_tab")
        self.horizontalLayout_2 = QHBoxLayout(self.predict_tab)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.predict_canvas_label = QLabel(self.predict_tab)
        self.predict_canvas_label.setObjectName(u"predict_canvas_label")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.predict_canvas_label.sizePolicy().hasHeightForWidth())
        self.predict_canvas_label.setSizePolicy(sizePolicy1)
        self.predict_canvas_label.setMinimumSize(QSize(384, 384))
        self.predict_canvas_label.setMaximumSize(QSize(384, 384))
        self.predict_canvas_label.setCursor(QCursor(Qt.CrossCursor))
        self.predict_canvas_label.setStyleSheet(u"QLabel{ background-color : black; color : white; }")

        self.horizontalLayout_2.addWidget(self.predict_canvas_label)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.model_selector = QComboBox(self.predict_tab)
        self.model_selector.setObjectName(u"model_selector")

        self.verticalLayout_2.addWidget(self.model_selector)

        self.prediction_probability_list = QListWidget(self.predict_tab)
        self.prediction_probability_list.setObjectName(u"prediction_probability_list")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.prediction_probability_list.sizePolicy().hasHeightForWidth())
        self.prediction_probability_list.setSizePolicy(sizePolicy2)

        self.verticalLayout_2.addWidget(self.prediction_probability_list)


        self.horizontalLayout_2.addLayout(self.verticalLayout_2)

        self.mode_selector_tab.addTab(self.predict_tab, "")
        self.train_tab = QWidget()
        self.train_tab.setObjectName(u"train_tab")
        self.mode_selector_tab.addTab(self.train_tab, "")
        self.draw_tab = QWidget()
        self.draw_tab.setObjectName(u"draw_tab")
        self.horizontalLayout_3 = QHBoxLayout(self.draw_tab)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.draw_canvas_label = QLabel(self.draw_tab)
        self.draw_canvas_label.setObjectName(u"draw_canvas_label")
        sizePolicy1.setHeightForWidth(self.draw_canvas_label.sizePolicy().hasHeightForWidth())
        self.draw_canvas_label.setSizePolicy(sizePolicy1)
        self.draw_canvas_label.setMinimumSize(QSize(384, 384))
        self.draw_canvas_label.setMaximumSize(QSize(384, 384))
        self.draw_canvas_label.setCursor(QCursor(Qt.CrossCursor))
        self.draw_canvas_label.setStyleSheet(u"QLabel{ background-color : black; }")

        self.horizontalLayout_3.addWidget(self.draw_canvas_label)

        self.symbols_to_draw_list = QListWidget(self.draw_tab)
        self.symbols_to_draw_list.setObjectName(u"symbols_to_draw_list")

        self.horizontalLayout_3.addWidget(self.symbols_to_draw_list)

        self.mode_selector_tab.addTab(self.draw_tab, "")

        self.verticalLayout.addWidget(self.mode_selector_tab)

        gyarte.setCentralWidget(self.centralwidget)

        self.retranslateUi(gyarte)

        self.mode_selector_tab.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(gyarte)
    # setupUi

    def retranslateUi(self, gyarte):
        gyarte.setWindowTitle(QCoreApplication.translate("gyarte", u"Gymnasie Arbete AI", None))
        self.predict_canvas_label.setText("")
        self.mode_selector_tab.setTabText(self.mode_selector_tab.indexOf(self.predict_tab), QCoreApplication.translate("gyarte", u"Predict", None))
#if QT_CONFIG(tooltip)
        self.mode_selector_tab.setTabToolTip(self.mode_selector_tab.indexOf(self.predict_tab), QCoreApplication.translate("gyarte", u"Draw an image and let the AI guess", None))
#endif // QT_CONFIG(tooltip)
        self.mode_selector_tab.setTabText(self.mode_selector_tab.indexOf(self.train_tab), QCoreApplication.translate("gyarte", u"Train", None))
#if QT_CONFIG(tooltip)
        self.mode_selector_tab.setTabToolTip(self.mode_selector_tab.indexOf(self.train_tab), QCoreApplication.translate("gyarte", u"Create and train a new or existing model", None))
#endif // QT_CONFIG(tooltip)
        self.draw_canvas_label.setText("")
        self.mode_selector_tab.setTabText(self.mode_selector_tab.indexOf(self.draw_tab), QCoreApplication.translate("gyarte", u"Draw (dev)", None))
#if QT_CONFIG(tooltip)
        self.mode_selector_tab.setTabToolTip(self.mode_selector_tab.indexOf(self.draw_tab), QCoreApplication.translate("gyarte", u"Draw a series of images which will be used for testing", None))
#endif // QT_CONFIG(tooltip)
    # retranslateUi

