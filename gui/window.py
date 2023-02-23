

################################################################################
## Form generated from reading UI file 'gui.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QHBoxLayout,
    QLabel, QLayout, QListWidget, QListWidgetItem,
    QMainWindow, QProgressBar, QSizePolicy, QTabWidget,
    QVBoxLayout, QWidget)

class Ui_MainWindow:
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(538, 432)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMaximumSize(QSize(538, 432))
        MainWindow.setTabShape(QTabWidget.TabShape.Rounded)
        self.centralwidget = QWidget(MainWindow)
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
        self.mode_selector_tab.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.mode_selector_tab.setStyleSheet(u"")
        self.mode_selector_tab.setTabPosition(QTabWidget.TabPosition.South)
        self.mode_selector_tab.setMovable(True)
        self.mode_selector_tab.setTabBarAutoHide(True)
        self.predict_tab = QWidget()
        self.predict_tab.setObjectName(u"predict_tab")
        self.horizontalLayout_2 = QHBoxLayout(self.predict_tab)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.predict_canvas_label = QLabel(self.predict_tab)
        self.predict_canvas_label.setObjectName(u"predict_canvas_label")
        sizePolicy.setHeightForWidth(self.predict_canvas_label.sizePolicy().hasHeightForWidth())
        self.predict_canvas_label.setSizePolicy(sizePolicy)
        self.predict_canvas_label.setMinimumSize(QSize(384, 384))
        self.predict_canvas_label.setMargin(0)

        self.horizontalLayout_2.addWidget(self.predict_canvas_label)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setSpacing(3)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.comboBox = QComboBox(self.predict_tab)
        self.comboBox.setObjectName(u"comboBox")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy1)
        self.comboBox.setMinimumSize(QSize(120, 0))
        self.comboBox.setMaximumSize(QSize(120, 16777215))

        self.verticalLayout_2.addWidget(self.comboBox)

        self.listWidget = QListWidget(self.predict_tab)
        self.listWidget.setObjectName(u"listWidget")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.listWidget.sizePolicy().hasHeightForWidth())
        self.listWidget.setSizePolicy(sizePolicy2)
        self.listWidget.setMaximumSize(QSize(120, 16777215))

        self.verticalLayout_2.addWidget(self.listWidget)


        self.horizontalLayout.addLayout(self.verticalLayout_2)


        self.horizontalLayout_2.addLayout(self.horizontalLayout)

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
        sizePolicy.setHeightForWidth(self.draw_canvas_label.sizePolicy().hasHeightForWidth())
        self.draw_canvas_label.setSizePolicy(sizePolicy)
        self.draw_canvas_label.setMinimumSize(QSize(384, 384))
        self.draw_canvas_label.setFrameShape(QFrame.Shape.NoFrame)
        self.draw_canvas_label.setFrameShadow(QFrame.Shadow.Plain)
        self.draw_canvas_label.setScaledContents(False)

        self.horizontalLayout_3.addWidget(self.draw_canvas_label)

        self.progressBar = QProgressBar(self.draw_tab)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setValue(24)
        self.progressBar.setOrientation(Qt.Orientation.Vertical)

        self.horizontalLayout_3.addWidget(self.progressBar)

        self.mode_selector_tab.addTab(self.draw_tab, "")

        self.verticalLayout.addWidget(self.mode_selector_tab)

        MainWindow.setCentralWidget(self.centralwidget)

        self.mode_selector_tab.setCurrentIndex(0)

        QMetaObject.connectSlotsByName(MainWindow)
