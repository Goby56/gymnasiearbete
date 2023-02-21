import sys
from PySide6 import QtCore, QtGui, QtWidgets

from PIL import Image, ImageDraw, ImageQt

# from ui import Ui_MainWindow

class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setGeometry(300, 300, 512, 512)
        self.setWindowTitle("Gymnasie Arbete AI")
        self.canvas = Canvas(self)
        self.show()

class Canvas(QtWidgets.QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setGeometry(0, 0, 140, 140)
        self.setText("")

        self.pixels = Image.new("RGB", (128, 128))
        self.update()
        self.draw = ImageDraw.Draw(self.pixels)
    
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        s = 1
        x, y = event.scenePosition().x() / 5, event.scenePosition().y() / 5
        print(x, y)
        self.draw.ellipse((x-s, y-s, x+s, y+s), fill="white", outline="white")
        self.update()

    def update(self):
        qt_img = ImageQt.ImageQt(self.pixels)
        pixmap = QtGui.QPixmap.fromImage(qt_img)
        pixmap = pixmap.scaled(140, 140)
        self.setPixmap(pixmap)
        self.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec())