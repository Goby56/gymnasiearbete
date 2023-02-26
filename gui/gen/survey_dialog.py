# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'survey_dialog.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLineEdit,
    QPushButton, QSizePolicy, QSpacerItem, QVBoxLayout,
    QWidget)

class Ui_survey_dialog(object):
    def setupUi(self, survey_dialog):
        if not survey_dialog.objectName():
            survey_dialog.setObjectName(u"survey_dialog")
        survey_dialog.resize(302, 140)
        self.verticalLayout = QVBoxLayout(survey_dialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer)

        self.participant_name_line_edit = QLineEdit(survey_dialog)
        self.participant_name_line_edit.setObjectName(u"participant_name_line_edit")

        self.verticalLayout_3.addWidget(self.participant_name_line_edit)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_2)


        self.horizontalLayout.addLayout(self.verticalLayout_3)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer_3)

        self.select_directory_button = QPushButton(survey_dialog)
        self.select_directory_button.setObjectName(u"select_directory_button")

        self.verticalLayout_2.addWidget(self.select_directory_button)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer_4)


        self.horizontalLayout.addLayout(self.verticalLayout_2)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.submit_config_button = QPushButton(survey_dialog)
        self.submit_config_button.setObjectName(u"submit_config_button")

        self.verticalLayout.addWidget(self.submit_config_button)


        self.retranslateUi(survey_dialog)

        QMetaObject.connectSlotsByName(survey_dialog)
    # setupUi

    def retranslateUi(self, survey_dialog):
        survey_dialog.setWindowTitle(QCoreApplication.translate("survey_dialog", u"Dialog", None))
        self.participant_name_line_edit.setPlaceholderText(QCoreApplication.translate("survey_dialog", u"Name of participant", None))
        self.select_directory_button.setText(QCoreApplication.translate("survey_dialog", u"Choose directory", None))
        self.submit_config_button.setText(QCoreApplication.translate("survey_dialog", u"Submit", None))
    # retranslateUi

