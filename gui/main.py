import sys, cv2, os, random, glob
import time
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QEvent, QObject, Qt
from PIL import Image, ImageDraw, ImageFilter, ImageQt
from typing import List, NamedTuple, Union
import collections

from gen.main_window import Ui_main_window
from gen.survey_dialog import Ui_survey_dialog

SYMBOL_MAPPINGS = "AaBbCDdEeFfGgHhIJKkLMNnOPQqRrSTtUVWXYZ0123456789"
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
    
    def set_image(self, label: QtWidgets.QLabel, image: Image):
        self.pixels = image
        self.set_pixmap(label)

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
        self.gui = Ui_main_window()
        self.gui.setupUi(self)
        self.current_tab = 0
        self.canvas = Canvas()
        

        self.survey_participant = ""
        self.survey_images = []
        self.current_survey_image = 0


        self.gui.symbols_to_draw_list.addItems(random.sample(SYMBOL_MAPPINGS, 25))

        self.load_models(blacklist=["test_plot"])

    @property
    def selected_model(self):
        return "test_adam"  # fixa så man kan välja modell i combobox. modelerna finns som strings i self.models.keys()

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
        guesses = self.get_prediction(self.selected_model)
        fomatted_guesses = [f"{p[0]}\t{p[1]*100:.0f}%" for p in guesses] # är det möjligt att göra lådan lite lite bredare så det inte behövs en vertical scroll?
        self.gui.prediction_probability_list.clear()
        self.gui.prediction_probability_list.addItems(fomatted_guesses)

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
        return [("N/A", i) for i in sorted(out_vec, reverse=True)]

    #endregion

    #region -x-x-x-x-x-x-x-x-x- Tab : AI Train -x-x-x-x-x-x-x-x-x-

    def train_ai(self, callbacks: dict, dataset_options: dict) -> None:
        """
        Trains the selected ai.

        Args:
            (callbacks) A dict of callback functions. Must have keys 
                "callback_training", "callback_validation", and "callback_epoch".\n
                The functions should look like the following:
                callback_training(summary: collections.namedtuple) -> None
                callback_validation(summary: collections.namedtuple) -> None
                callback_epoch(epoch: int) -> None
            
            (dataset_options) The dataset keywords excluding dataset_augmentaion. 
                "image_size": tuple[int, int] must be included

        """
        ai = self.models[self.selected_model]
        assert "image_size" in dataset_options, "image_size must be included in dataset_options"

        dataset = sample.CompiledDataset(
            filename=ai.model.dataset,
            data_augmentation=ai.model.data_augmentation,
            **dataset_options
        )

        sample.train.train(
            network=ai.network,
            dataset=dataset,
            **callbacks
        )

    #endregion

    #region -x-x-x-x-x-x-x-x-x- Tab : Survey Guess -x-x-x-x-x-x-x-x-x-

    @EventHandler.onclick("survey_config_button")
    def open_survey_dialog(self, source: QObject, event: QEvent):
        def on_submit(name, path):
            print(path)
            # for p in glob.iglob(path+"*")
            
        self.survey_dialog = SurveyDialog(on_submit)
        # self.survey_dialog.setModal(True)
        self.survey_dialog.show()

    @EventHandler.onclick("next_image_button")
    @EventHandler.onclick("predict_canvas_label")
    def update_survey_image(self, source: QObject, event: QEvent):
        if source.objectName() == "next_image_button":

            self.gui.guess_canvas_label
            self.gui.predict_canvas_label


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
    def __init__(self, on_submit: callable) -> None:
        super().__init__()
        self.gui = Ui_survey_dialog()
        self.gui.setupUi(self)

        self.on_submit = on_submit
        self.folder_selection = ""
        self.participant_name = ""

    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        EventHandler.trigger(self, source, event)
        return QtWidgets.QDialog.eventFilter(self, source, event)

    @EventHandler.onclick("select_directory_button")
    def open_folder_selector(self, source: QObject, event: QEvent):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self)
        if folder:
            self.folder_selection = folder

    @EventHandler.onclick("submit_config_button")
    def submit_config(self, source: QObject, event: QEvent):
        print(type(self.gui))
        self.participant_name = self.gui.participant_name_line_edit.text()
        print(self.participant_name, self.folder_selection)
        if self.folder_selection and self.participant_name:
            self.on_submit(self.participant_name, self.folder_selection)
        else:
            # Raise error if not enough information was provided
            pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Window()
    win.load_models()
    win.show()
    app.installEventFilter(win)
    sys.exit(app.exec())