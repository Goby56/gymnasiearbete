# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
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
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QProgressBar, QPushButton, QSizePolicy, QSpacerItem,
    QTabWidget, QVBoxLayout, QWidget)

class Ui_main_window(object):
    def setupUi(self, main_window):
        if not main_window.objectName():
            main_window.setObjectName(u"main_window")
        main_window.resize(538, 432)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(main_window.sizePolicy().hasHeightForWidth())
        main_window.setSizePolicy(sizePolicy)
        main_window.setMaximumSize(QSize(538, 432))
        main_window.setTabShape(QTabWidget.Rounded)
        self.centralwidget = QWidget(main_window)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy1)
        self.centralwidget.setMaximumSize(QSize(538, 432))
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
        self.mode_selector_tab.setMovable(False)
        self.mode_selector_tab.setTabBarAutoHide(True)
        self.predict_tab = QWidget()
        self.predict_tab.setObjectName(u"predict_tab")
        self.horizontalLayout_2 = QHBoxLayout(self.predict_tab)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.predict_canvas_label = QLabel(self.predict_tab)
        self.predict_canvas_label.setObjectName(u"predict_canvas_label")
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.predict_canvas_label.sizePolicy().hasHeightForWidth())
        self.predict_canvas_label.setSizePolicy(sizePolicy2)
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
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.prediction_probability_list.sizePolicy().hasHeightForWidth())
        self.prediction_probability_list.setSizePolicy(sizePolicy3)
        font = QFont()
        font.setFamilies([u"Consolas"])
        font.setPointSize(18)
        self.prediction_probability_list.setFont(font)

        self.verticalLayout_2.addWidget(self.prediction_probability_list)


        self.horizontalLayout_2.addLayout(self.verticalLayout_2)

        self.mode_selector_tab.addTab(self.predict_tab, "")
        self.train_tab = QWidget()
        self.train_tab.setObjectName(u"train_tab")
        self.verticalLayout_5 = QVBoxLayout(self.train_tab)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.toggle_graph_toolbar = QPushButton(self.train_tab)
        self.toggle_graph_toolbar.setObjectName(u"toggle_graph_toolbar")
        self.toggle_graph_toolbar.setMaximumSize(QSize(25, 16777215))

        self.horizontalLayout_4.addWidget(self.toggle_graph_toolbar)

        self.configure_training_button = QPushButton(self.train_tab)
        self.configure_training_button.setObjectName(u"configure_training_button")

        self.horizontalLayout_4.addWidget(self.configure_training_button)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.start_stop_training_button = QPushButton(self.train_tab)
        self.start_stop_training_button.setObjectName(u"start_stop_training_button")

        self.horizontalLayout_5.addWidget(self.start_stop_training_button)

        self.pause_resume_training_button = QPushButton(self.train_tab)
        self.pause_resume_training_button.setObjectName(u"pause_resume_training_button")

        self.horizontalLayout_5.addWidget(self.pause_resume_training_button)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_5)


        self.verticalLayout_5.addLayout(self.horizontalLayout_4)

        self.mode_selector_tab.addTab(self.train_tab, "")
        self.guess_tab = QWidget()
        self.guess_tab.setObjectName(u"guess_tab")
        self.horizontalLayout = QHBoxLayout(self.guess_tab)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.guess_canvas_label = QLabel(self.guess_tab)
        self.guess_canvas_label.setObjectName(u"guess_canvas_label")
        sizePolicy2.setHeightForWidth(self.guess_canvas_label.sizePolicy().hasHeightForWidth())
        self.guess_canvas_label.setSizePolicy(sizePolicy2)
        self.guess_canvas_label.setMinimumSize(QSize(384, 384))
        self.guess_canvas_label.setMaximumSize(QSize(384, 384))
        self.guess_canvas_label.setStyleSheet(u"QLabel{ background-color : black; }")

        self.horizontalLayout.addWidget(self.guess_canvas_label)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.survey_config_button = QPushButton(self.guess_tab)
        self.survey_config_button.setObjectName(u"survey_config_button")

        self.verticalLayout_3.addWidget(self.survey_config_button)

        self.image_guess_input_line = QLineEdit(self.guess_tab)
        self.image_guess_input_line.setObjectName(u"image_guess_input_line")
        self.image_guess_input_line.setMinimumSize(QSize(120, 120))
        self.image_guess_input_line.setMaximumSize(QSize(130, 130))
        font1 = QFont()
        font1.setFamilies([u"Consolas"])
        font1.setPointSize(72)
        self.image_guess_input_line.setFont(font1)
        self.image_guess_input_line.setAutoFillBackground(False)

        self.verticalLayout_3.addWidget(self.image_guess_input_line)

        self.next_image_button = QPushButton(self.guess_tab)
        self.next_image_button.setObjectName(u"next_image_button")

        self.verticalLayout_3.addWidget(self.next_image_button)

        self.previous_image_button = QPushButton(self.guess_tab)
        self.previous_image_button.setObjectName(u"previous_image_button")

        self.verticalLayout_3.addWidget(self.previous_image_button)

        self.images_left_progress_bar = QProgressBar(self.guess_tab)
        self.images_left_progress_bar.setObjectName(u"images_left_progress_bar")
        self.images_left_progress_bar.setValue(24)
        self.images_left_progress_bar.setOrientation(Qt.Horizontal)

        self.verticalLayout_3.addWidget(self.images_left_progress_bar)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer)


        self.horizontalLayout.addLayout(self.verticalLayout_3)

        self.mode_selector_tab.addTab(self.guess_tab, "")
        self.draw_tab = QWidget()
        self.draw_tab.setObjectName(u"draw_tab")
        self.horizontalLayout_3 = QHBoxLayout(self.draw_tab)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.draw_canvas_label = QLabel(self.draw_tab)
        self.draw_canvas_label.setObjectName(u"draw_canvas_label")
        sizePolicy2.setHeightForWidth(self.draw_canvas_label.sizePolicy().hasHeightForWidth())
        self.draw_canvas_label.setSizePolicy(sizePolicy2)
        self.draw_canvas_label.setMinimumSize(QSize(384, 384))
        self.draw_canvas_label.setMaximumSize(QSize(384, 384))
        self.draw_canvas_label.setCursor(QCursor(Qt.CrossCursor))
        self.draw_canvas_label.setStyleSheet(u"QLabel{ background-color : black; }")

        self.horizontalLayout_3.addWidget(self.draw_canvas_label)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.survey_add_config_button = QPushButton(self.draw_tab)
        self.survey_add_config_button.setObjectName(u"survey_add_config_button")

        self.verticalLayout_4.addWidget(self.survey_add_config_button)

        self.symbols_to_draw_list = QListWidget(self.draw_tab)
        self.symbols_to_draw_list.setObjectName(u"symbols_to_draw_list")
        sizePolicy2.setHeightForWidth(self.symbols_to_draw_list.sizePolicy().hasHeightForWidth())
        self.symbols_to_draw_list.setSizePolicy(sizePolicy2)
        self.symbols_to_draw_list.setFont(font1)

        self.verticalLayout_4.addWidget(self.symbols_to_draw_list)

        self.start_stop_survey_button = QPushButton(self.draw_tab)
        self.start_stop_survey_button.setObjectName(u"start_stop_survey_button")
        self.start_stop_survey_button.setEnabled(True)

        self.verticalLayout_4.addWidget(self.start_stop_survey_button)


        self.horizontalLayout_3.addLayout(self.verticalLayout_4)

        self.mode_selector_tab.addTab(self.draw_tab, "")

        self.verticalLayout.addWidget(self.mode_selector_tab)

        main_window.setCentralWidget(self.centralwidget)

        self.retranslateUi(main_window)

        self.mode_selector_tab.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(main_window)
    # setupUi

    def retranslateUi(self, main_window):
        main_window.setWindowTitle(QCoreApplication.translate("main_window", u"Gymnasie Arbete AI", None))
        self.predict_canvas_label.setText("")
        self.mode_selector_tab.setTabText(self.mode_selector_tab.indexOf(self.predict_tab), QCoreApplication.translate("main_window", u"Use model", None))
#if QT_CONFIG(tooltip)
        self.mode_selector_tab.setTabToolTip(self.mode_selector_tab.indexOf(self.predict_tab), QCoreApplication.translate("main_window", u"Draw an image and let the AI guess", None))
#endif // QT_CONFIG(tooltip)
        self.toggle_graph_toolbar.setText(QCoreApplication.translate("main_window", u"\u25bc", None))
        self.configure_training_button.setText(QCoreApplication.translate("main_window", u"Configure", None))
        self.start_stop_training_button.setText(QCoreApplication.translate("main_window", u"Start", None))
        self.pause_resume_training_button.setText(QCoreApplication.translate("main_window", u"Pause", None))
        self.mode_selector_tab.setTabText(self.mode_selector_tab.indexOf(self.train_tab), QCoreApplication.translate("main_window", u"Train model", None))
#if QT_CONFIG(tooltip)
        self.mode_selector_tab.setTabToolTip(self.mode_selector_tab.indexOf(self.train_tab), QCoreApplication.translate("main_window", u"Create and train a new or existing model", None))
#endif // QT_CONFIG(tooltip)
        self.guess_canvas_label.setText("")
        self.survey_config_button.setText(QCoreApplication.translate("main_window", u"Configure", None))
        self.image_guess_input_line.setText("")
        self.next_image_button.setText(QCoreApplication.translate("main_window", u"Next Image", None))
        self.previous_image_button.setText(QCoreApplication.translate("main_window", u"Previous Image", None))
        self.mode_selector_tab.setTabText(self.mode_selector_tab.indexOf(self.guess_tab), QCoreApplication.translate("main_window", u"Survey", None))
        self.draw_canvas_label.setText("")
        self.survey_add_config_button.setText(QCoreApplication.translate("main_window", u"Configure", None))
        self.start_stop_survey_button.setText(QCoreApplication.translate("main_window", u"Start", None))
        self.mode_selector_tab.setTabText(self.mode_selector_tab.indexOf(self.draw_tab), QCoreApplication.translate("main_window", u"Add survey samples", None))
#if QT_CONFIG(tooltip)
        self.mode_selector_tab.setTabToolTip(self.mode_selector_tab.indexOf(self.draw_tab), QCoreApplication.translate("main_window", u"Draw a series of images which will be used for testing", None))
#endif // QT_CONFIG(tooltip)
    # retranslateUi

