import sys, cv2, os, random
import time
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QEvent, QObject, Qt
from PIL import Image, ImageDraw, ImageFilter, ImageQt
from typing import List, NamedTuple, Union
import collections

from gui_src import Ui_gyarte
from config_dialog_src import Ui_survey_config_dialog

symbol_mappings = "AaBbCDdEeFfGgHhIJKkLMNnOPQqRrSTtUVWXYZ0123456789"
SURVEY_PATH = os.path.join(os.getcwd(), "gui", "survey_images")

sys.path.append(os.getcwd())
PATH_MODELS = os.path.join(os.getcwd(), "data\\models")

import sample

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
    
    @staticmethod
    def onclick(button_name: str):
        def dec(func):
            EventHandler.add(Event(func, QEvent.MouseButtonPress, [button_name], 
                                   None, [Qt.MouseButton.LeftButton], None))
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
        
        self.gui.symbols_to_draw_list.addItems(random.sample(symbol_mappings, 25))

        self.load_models(blacklist=["test_plot"])

    def eventFilter(self, source: QObject, event: QEvent):
        EventHandler.trigger(self, source, event)
        return QtWidgets.QMainWindow.eventFilter(self, source, event)

    @EventHandler.on(QEvent.Paint, from_widgets=["mode_selector_tab"])
    def update_tab_index(self, source: QObject, event: QEvent):
        self.current_tab = source.currentIndex()

    @EventHandler.on(QEvent.MouseButtonPress, mbuttons=[Qt.MouseButton.LeftButton], 
                     from_widgets=["draw_canvas_label", "predict_canvas_label"])
    @EventHandler.on(QEvent.MouseMove, mbuttons=[Qt.MouseButton.LeftButton], 
                     from_widgets=["draw_canvas_label", "predict_canvas_label"])
    def draw_on_canvas(self, source: QObject, event: QEvent):
        x, y = event.position().x() / self.canvas.scaler, event.position().y() / self.canvas.scaler
        self.canvas.draw(x, y, source)

    @EventHandler.on(QEvent.MouseButtonPress, mbuttons=[Qt.MouseButton.RightButton], 
                     from_widgets=["draw_canvas_label", "predict_canvas_label"])
    def reset_canvas(self, source: QObject, event: QEvent):
        self.canvas.erease(source)

    #region -x-x-x-x-x-x-x-x-x- Tab : AI Predict -x-x-x-x-x-x-x-x-x-

    @EventHandler.on(QEvent.MouseButtonRelease, from_widgets=["predict_canvas_label"], 
                     mbuttons=[Qt.MouseButton.LeftButton])
    def display_prediction(self, source: QObject, event: QEvent):
        guesses = self.get_prediction("test_adam")
        print(guesses[:2])

    def load_models(self, blacklist: list[str] = []) -> None:
        """
        Loads all models from "./data/models" that aren't blacklisted.

        Args:
            (blacklist) A list of model names that are to be blacklisted. Set to empty as default.
        """
        model_names = os.listdir(PATH_MODELS)
        self.models = {}
        data = collections.namedtuple("data", ["network", "model"])

        for model_name in model_names:
            if not model_name in blacklist:
                model = sample.Model(name=model_name)
                network = sample.Network(model=model)
                self.models[model_name] = data(network, model)

    def get_prediction(self, model_name: str, standardize=True) -> list[tuple[str, int]]:
        """
        Does a forward pass with the given model and return a sorted list of probabilities.

        Args:
            (model_name) The model name as it apears in the "./data/models" folder.
            (standardize) If the canvas image should be standardized. Set to True as defualt.
        Returns:
            (sorted_probabilities) a sorted list of tupels in the form of (label: str, prob: int)
        """

        ai = self.models[model_name]
        image = self.canvas.downscaled.convert("L")
        in_vec = np.asarray(image).flatten()
        in_vec.shape = (1, len(in_vec))
        #in_vec = sample.CompiledDataset.standardize_image(in_vec) if standardize else in_vec / 255
        out_vec = ai.network.forward(in_vec).flatten()
        if ai.model.has_mapping:
            guess = list(zip(ai.model.mapping, out_vec))
            return sorted(guess, key=lambda x: x[1], reverse=True)
        return [("N/A", i) for i in np.sort(out_vec)]

    #endregion

    #region -x-x-x-x-x-x-x-x-x- Tab : AI Train -x-x-x-x-x-x-x-x-x-


    #endregion

    #region -x-x-x-x-x-x-x-x-x- Tab : Survey Guess -x-x-x-x-x-x-x-x-x-

    @EventHandler.onclick("survey_config_button")
    def open_survey_dialog(self, source: QObject, event: QEvent):
        self.survey_dialog = SurveyDialog()
        self.survey_dialog.show()

    #endregion
    
    #region -x-x-x-x-x-x-x-x-x- Tab : Survey Draw -x-x-x-x-x-x-x-x-x-

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

    #endregion

class SurveyDialog(QtWidgets.QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.gui = Ui_survey_config_dialog()
        self.gui.setupUi(self)

    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        EventHandler.trigger(self, source, event)
        return QtWidgets.QDialog.eventFilter(self, source, event)

    @EventHandler.onclick("survey_chooe_directory_button") # är det verkligen chooe?
    def open_folder_selector(self, source: QObject, event: QEvent):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self)
        if folder:
            print(folder)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Window()
    win.load_models()
    win.show()
    app.installEventFilter(win)
    sys.exit(app.exec())