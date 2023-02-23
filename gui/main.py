import sys
from PySide6 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageDraw, ImageQt

from window import Ui_MainWindow

class GUI(Ui_MainWindow):
    def __init__(self, main_window: QtWidgets.QMainWindow) -> None:
        self.setupUi(main_window)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    gui = GUI(win)
    win.show()
    sys.exit(app.exec())