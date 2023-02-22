import sys
from PySide6 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageDraw, ImageQt

from window import Ui_MainWindow

class GUI(Ui_MainWindow):
    def __init__(self, main_window: QtWidgets.QMainWindow) -> None:
        self.setupUi(main_window)


        main_window.show()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI(QtWidgets.QMainWindow())

    sys.exit(app.exec_())