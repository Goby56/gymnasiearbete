import sys, cv2, os, random
import time
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QEvent, QObject, Qt
from PIL import Image, ImageDraw, ImageFilter, ImageQt
from typing import List, NamedTuple, Union

from gui_src import Ui_gyarte

symbol_mappings = "AaBbCDdEeFfGgHhIJKkLMNnOPQqRrSTtUVWXYZ0123456789"
survey_file_path = os.path.join(os.getcwd(), "gui", "survey_images")

class Event(NamedTuple):
    callback: callable
    etype: QEvent
    widgets: List[str]
    keys: List[Qt.Key]
    mbuttons: List[Qt.MouseButton]
    tab: int

class EventHandler:
    listeners: List[Event] = []

    @classmethod
    def add(cls, e: Event):
        if e in cls.listeners: return
        cls.listeners.append(e)
    
    @classmethod
    def remove(cls, e: Event):
        if not e in cls.listeners: return
        cls.listeners.remove(e)

    @classmethod
    def trigger(cls, ref: object, source: QObject, event: QEvent):
        for callback, etype, widgets, keys, mbuttons, tab in cls.listeners:
            if event.type() != etype:
                continue
            if type(source) == type(ref):
                continue
            if widgets != None and source.objectName() not in widgets:
                continue
            if tab != None and tab != ref.current_tab:
                continue
            
            if event.type() == QEvent.KeyPress: 
                if type(source) != QtGui.QWindow:
                    continue
                if keys != None and event.key() in keys:
                    callback(ref, source, event)
                continue
            if event.type() == QEvent.MouseButtonPress:
                if mbuttons != None and event.buttons() in mbuttons:
                    callback(ref, source, event)
                continue
            if event.type() == QEvent.MouseMove:
                if mbuttons != None and event.buttons() in mbuttons:
                    callback(ref, source, event)
                continue
            callback(ref, source, event)

    @staticmethod
    def on(event_type: QEvent, from_widgets: List[str] = None, keys: List[Qt.Key] = None, 
           mbuttons: List[Qt.MouseButton] = None, tab=None):
        def dec(func):
            EventHandler.add(Event(func, event_type, from_widgets, keys, mbuttons, tab))
            return func
        return dec
    
class Canvas:
    def __init__(self) -> None:
        self.pixels = Image.new("RGB", (128, 128))
        self.drawer = ImageDraw.Draw(self.pixels)
        self.scaler = 3

    @property
    def downscaled(self):
        blurred = self.pixels.filter(ImageFilter.GaussianBlur(radius=1))
        arr1 = np.array(blurred)
        cv_img = cv2.cvtColor(arr1, cv2.COLOR_RGB2BGR)
        downsampled = cv2.resize(cv_img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        arr2 = cv2.cvtColor(downsampled, cv2.COLOR_BGR2RGB)
        return Image.fromarray(arr2) # Pil img

    def draw(self, x, y, label: QtWidgets.QLabel):
        size = 3
        self.drawer.ellipse((x-size, y-size, x+size, y+size), fill="white", outline="white")
        self.set_pixmap(label)

    def erease(self, label: QtWidgets.QLabel):
        self.drawer.rectangle((0, 0, 128, 128), fill="black", outline="black")
        self.set_pixmap(label)

    def set_pixmap(self, label: QtWidgets.QLabel):
        qt_img = ImageQt.ImageQt(self.downscaled)
        pixmap = QtGui.QPixmap.fromImage(qt_img)
        pixmap = pixmap.scaled(128*self.scaler, 128*self.scaler)
        label.setPixmap(pixmap)

class Window(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.gui = Ui_gyarte()
        self.gui.setupUi(self)
        self.current_tab = 0
        self.canvas = Canvas()
        
        self.event_handler = EventHandler()
        self.gui.symbols_to_draw_list.addItems(random.sample(symbol_mappings, 25))

    def eventFilter(self, source: QObject, event: QEvent):
        self.event_handler.trigger(self, source, event)
        return QtWidgets.QMainWindow.eventFilter(self, source, event)
    
    @EventHandler.on(QEvent.Paint, from_widgets=["mode_selector_tab"])
    def update_tab_index(self, source: QObject, event: QEvent):
        self.current_tab = source.currentIndex()

    @EventHandler.on(QEvent.MouseButtonPress, mbuttons=[Qt.MouseButton.LeftButton], from_widgets=["draw_canvas_label", "predict_canvas_label"])
    @EventHandler.on(QEvent.MouseMove, mbuttons=[Qt.MouseButton.LeftButton], from_widgets=["draw_canvas_label", "predict_canvas_label"])
    def draw_on_canvas(self, source: QObject, event: QEvent):
        x, y = event.position().x() / self.canvas.scaler, event.position().y() / self.canvas.scaler
        self.canvas.draw(x, y, source)

    @EventHandler.on(QEvent.MouseButtonPress, mbuttons=[Qt.MouseButton.RightButton], from_widgets=["draw_canvas_label", "predict_canvas_label"])
    def reset_canvas(self, source: QObject, event: QEvent):
        self.canvas.erease(source)

    @EventHandler.on(QEvent.KeyPress, keys=[Qt.Key.Key_Return], tab=3)
    def save_survey_image(self, source: QObject, event: QEvent):
        if event.isAutoRepeat():
            return
        # img = downscale_img(self.pixels)
        # symbol_drawn = self.gui.symbols_to_draw_list.takeItem(0).text()
        # if symbol_drawn.isdigit():
        #     prefix = "nu"
        # elif symbol_drawn.isupper():
        #     prefix = "up"
        # elif symbol_drawn.islower():
        #     prefix = "lo"
        # drawer = "oscar"
        # file_name = f"{drawer}_{prefix}_{symbol_drawn}.png"
        # file_path = os.path.join(survey_file_path, file_name)
        # img.save(file_path, "PNG")
        # self.reset_canvas(self.gui.draw_canvas_label)
    
    @EventHandler.on(QEvent.KeyPress, keys=[Qt.Key.Key_O], tab=2)
    def get_folder(self, source: QObject, event: QEvent):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self)
        if folder:
            print(folder)

    @EventHandler.on(QEvent.MouseMove, )
    def display_prediction(self, source: QObject, event: QEvent):
        pass

    def get_prediction(self):
        img = self.canvas.downscaled


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Window()
    win.show()
    app.installEventFilter(win)
    sys.exit(app.exec())