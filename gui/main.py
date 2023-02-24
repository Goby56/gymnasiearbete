import sys, cv2, os, random
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageDraw, ImageFilter, ImageQt

from gui_src import Ui_gyarte

def downscale_img(pil_img: Image.Image):
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
    arr1 = np.array(blurred)
    cv_img = cv2.cvtColor(arr1, cv2.COLOR_RGB2BGR)
    downsampled = cv2.resize(cv_img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    arr2 = cv2.cvtColor(downsampled, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(arr2)
    return pil_img

symbol_mappings = "AaBbCDdEeFfGgHhIJKkLMNnOPQqRrSTtUVWXYZ0123456789"
survey_file_path = os.path.join(os.getcwd(), "gui", "survey_images")

class Window(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.gui = Ui_gyarte()
        self.gui.setupUi(self)

        self.canvas_scaler = 3
        self.pixels = Image.new("RGB", (128, 128))
        self.draw = ImageDraw.Draw(self.pixels)


        self.gui.symbols_to_draw_list.addItems(random.sample(symbol_mappings, 25))

        self.current_tab_mode = 0

    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent):
        # TODO implement decorators to specify which method handles which event
        # ----- Handle saving image -----
        if event.type() == QtCore.QEvent.KeyPress:
            self.grabKeyboard()
            key = QtCore.Qt.Key(event.key())
            if key == QtCore.Qt.Key.Key_Return and not event.isAutoRepeat() and source == self:
                img = downscale_img(self.pixels)
                symbol_drawn = self.gui.symbols_to_draw_list.takeItem(0).text()
                file_path = os.path.join(survey_file_path, symbol_drawn)
                img.save(file_path + ".png", "PNG")

        # ----- Handle drawing -----
        if source.objectName() in ["draw_canvas_label", "predict_canvas_label"]:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                if event.buttons() == QtCore.Qt.MouseButton.RightButton:
                    self.reset_canvas(source)
                if event.buttons() == QtCore.Qt.MouseButton.LeftButton:
                    self.canvas_draw(event.position(), source)
            if event.type() == QtCore.QEvent.MouseMove and event.buttons() == QtCore.Qt.MouseButton.LeftButton:
                self.canvas_draw(event.position(), source)
        return 0

    def default_event(self, source, event):
        return QtWidgets.QMainWindow.eventFilter(self, source, event)

    def canvas_draw(self, event_pos, source):
        s = 4
        x, y = event_pos.x() / self.canvas_scaler, event_pos.y() / self.canvas_scaler
        self.draw.ellipse((x-s, y-s, x+s, y+s), fill="white", outline="white")
        self.set_pixmap(source)

    def reset_canvas(self, source):
        self.draw.rectangle((0, 0, 128, 128), fill="black", outline="black")
        self.set_pixmap(source)

    def set_pixmap(self, label: QtWidgets.QLabel):
        pil_img = downscale_img(self.pixels)
        qt_img = ImageQt.ImageQt(pil_img)
        pixmap = QtGui.QPixmap.fromImage(qt_img)
        pixmap = pixmap.scaled(128*self.canvas_scaler, 128*self.canvas_scaler)
        label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Window()
    win.show()
    app.installEventFilter(win)
    sys.exit(app.exec())