import sys, cv2, os, random, glob, json, re, collections, matplotlib, time
import threading
import numpy as np
from scipy.interpolate import BPoly

from PIL import Image, ImageDraw, ImageFilter, ImageQt
from typing import List, NamedTuple, Union

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QEvent, QObject, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure as MatplotFigure

from gen.main_window import Ui_main_window
from gen.survey_dialog import Ui_survey_dialog

sys.path.append(os.getcwd())
SURVEY_IMAGES_PATH = os.path.join(os.getcwd(), "survey\\images")
EMNIST_DATASET_PATH = os.path.join(os.getcwd(), "data\\EMNIST")
ANALYSIS_DIGITS_PATH = os.path.join(os.getcwd(), "analysis\\digit_images")
SURVEY_GUESSES_PATH = os.path.join(os.getcwd(), "survey\\guesses")
MODELS_PATH = os.path.join(os.getcwd(), "data\\models")
TRAINING_GRAPHS_PATH = os.path.join(os.getcwd(), "analysis")

import sample
from sample.train import Summary

class Session(NamedTuple):
    count: int
    epoch: int
    excess: int
    steps_per_epoch: int

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
            if callback.__qualname__.split(".")[0] != ref.__class__.__name__:
                continue
            if event.type() != etype:
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

    def set_pixmap(self, label: QtWidgets.QLabel, image: Image = None):
        pil_img = self.downscaled if image == None else image
        qt_img = ImageQt.ImageQt(pil_img)
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

        # AI PREDICT
        # self.load_models(blacklist=["test_plot"]) 
        self.load_models(blacklist=[".templates"])

        # AI TRAIN
        self.model_to_train = None
        self.training_state = "" # can be continue, pause or stop

        self.graph = MatplotFigure()
        self.graph.set_tight_layout(True)
        self.gui.plot_canvas = FigureCanvasQTAgg(self.graph)
        self.gui.plot_toolbar = NavigationToolbar2QT(self.gui.plot_canvas)
        self.gui.verticalLayout_5.addWidget(self.gui.plot_canvas)
        self.gui.verticalLayout_5.addWidget(self.gui.plot_toolbar)
        self.gui.pause_resume_training_button.hide()
        self.gui.plot_toolbar.hide()

        self.t = []
        self.loss = []
        self.accuracy = []
        self.test_acc = []
        self.test_steps = []
        
        self.loss_ax = self.graph.add_subplot(111)
        self.loss_ax.set_ylabel("loss")
        self.loss_ax.set_xlabel("steps")
        self.loss_ax.set_ylim([0,5])
        self.acc_ax = self.loss_ax.twinx()
        self.acc_ax.set_ylabel("accuracy")
        self.acc_ax.set_ylim([0,1])

        self.loss_line, = self.loss_ax.plot(self.t, self.loss, label="loss", color="blue")
        self.acc_line, = self.acc_ax.plot(self.t, self.accuracy, label="accuracy", color="orange")
        self.test_acc_line, = self.acc_ax.plot(self.t, self.test_acc, label="test_accuracy", color="green")

        lines = [self.loss_line, self.acc_line, self.test_acc_line]
        labels = [l.get_label() for l in lines]
        self.loss_ax.legend(lines, labels, loc="center left")

        # SURVEY STUFF
        self.survey_participant = ""
        self.survey_images = self.load_images(SURVEY_IMAGES_PATH)
        random.seed(69)
        random.shuffle(self.survey_images)
        self.current_survey_image = 0
        self.canvas.set_pixmap(self.gui.guess_canvas_label, 
                              self.survey_images[self.current_survey_image])
        self.gui.images_left_progress_bar.setValue(0)

        # ADD SURVEY SAMPLES

    def eventFilter(self, source: QObject, event: QEvent):
        EventHandler.trigger(self, source, event)
        return QtWidgets.QMainWindow.eventFilter(self, source, event)
    
    def show_dialog(self, icon: QMessageBox.Icon, title: str, text: str, info_text: str = None):
        QtWidgets.QMessageBox.Icon.Warning
        msg = QtWidgets.QMessageBox()
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(text)
        if info_text != None:
            msg.setInformativeText(info_text)
        return msg.exec()

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

    @property
    def selected_model(self):
        return self.gui.model_selector.currentText()

    @EventHandler.on(QEvent.InputMethodQuery, from_widgets=["model_selector"])
    @EventHandler.on(QEvent.MouseButtonRelease, from_widgets=["predict_canvas_label"], 
                     mbuttons=[Qt.MouseButton.LeftButton])
    def display_prediction(self, source: QObject, event: QEvent):
        guesses = self.get_prediction(self.selected_model)
        fomatted_guesses = [f"{p[0]}:{p[1]*100:.0f}%" for p in guesses]
        self.gui.prediction_probability_list.clear()
        self.gui.prediction_probability_list.addItems(fomatted_guesses)

    def load_models(self, blacklist: list[str] = []) -> None:
        """
        Loads all models from "./data/models" that aren't blacklisted.

        Args:
            (blacklist) A list of model names that are to be blacklisted. Set to empty as default.
        """
        model_names = os.listdir(MODELS_PATH)
        self.models = {}
        data = collections.namedtuple("data", ["network", "model"])

        for model_name in model_names:
            if not model_name in blacklist:
                model = sample.Model(name=model_name)
                network = sample.Network(model=model)
                self.models[model_name] = data(network, model)
        self.gui.model_selector.clear()
        self.gui.model_selector.addItems(self.models.keys())

    def get_prediction(self, model_name: str) -> list[tuple[str, int]]:
        """
        Does a forward pass with the given model and return a sorted list of probabilities.

        Args:
            (model_name) The model name as it apears in the "./data/models" folder.
        Returns:
            (sorted_probabilities) a sorted list of tupels in the form of (label: str, prob: int)
        """

        ai = self.models[model_name]
        image = self.canvas.downscaled.convert("L")
        in_vec = np.asarray(image).flatten()
        in_vec.shape = (1, len(in_vec))
        out_vec = ai.network.forward(in_vec).flatten()
        guess = list(zip(ai.model.mapping, out_vec))
        return sorted(guess, key=lambda x: x[1], reverse=True)

    #endregion

    #region -x-x-x-x-x-x-x-x-x- Tab : AI Train -x-x-x-x-x-x-x-x-x-

    @EventHandler.onclick("toggle_graph_toolbar")
    def toggle_graph_toolbar(self, source: QObject, event: QEvent):
        if self.gui.plot_toolbar.isHidden():
            self.gui.plot_toolbar.show()
            source.setText("▲")
        else:
            self.gui.plot_toolbar.hide()
            source.setText("▼")

    @EventHandler.onclick("choose_model_to_train_button")
    def set_model_to_train(self, source: QObject, event: QEvent):
        f = QtWidgets.QFileDialog.getOpenFileName(self, dir=MODELS_PATH)[0]
        model_name = re.search("(?<=models/)\S+(?=\/config\.json)", f)
        if model_name:
            self.model_to_train = model_name.group()

    @EventHandler.onclick("start_stop_training_button")
    def handle_training(self, source: QObject, event: QEvent):
        if not self.model_to_train:
            self.show_dialog(QMessageBox.Icon.Warning, "Error", "No module selected")
            return
        if source.text() == "Start":
            self.training_state = "continue"
            options = {}
            self.training_thread = threading.Thread(target=self.train_ai, args=(self.plot_training, options))
            self.training_thread.start()
            source.setText("Stop")
            self.gui.pause_resume_training_button.show()
        elif source.text() == "Stop":
            self.training_state = "stop"
            self.gui.pause_resume_training_button.hide()
            self.gui.pause_resume_training_button.setText("Pause")
            source.setText("Start")
            self.smooth_test_acc()
            self.save_graph()
            self.reset_graph()

    @EventHandler.on(QEvent.Close)
    def stop_training(self, source: QObject, event: QEvent):
        self.training_state = "stop"

    def save_graph(self):
        model_path = TRAINING_GRAPHS_PATH+f"\\{self.model_to_train}"
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        sessions = list(filter(lambda fname: re.search("[0-9]+(-[0-9]+\.[0-9]{1,2}){3}(?=\.png$)", fname), 
                               os.listdir(model_path)))
        
        epoch = self.prev_summary.epoch + self.prev_summary.step / self.prev_summary.steps_per_epoch
        acc = self.accuracy[-1]
        tacc = self.test_acc[-1]
        fname = f"{len(sessions)+1}-{epoch:.2f}-{acc:.2f}-{tacc:.2f}.png"
        path = TRAINING_GRAPHS_PATH+f"\\{self.model_to_train}\\{fname}"

        self.loss_ax.set_title(f"epoch {epoch:.2f}")
        self.graph.savefig(path)

    @EventHandler.onclick("pause_resume_training_button")
    def toggle_training(self, source: QObject, event: QEvent):
        if source.text() == "Pause":
            self.training_state = "pause"
            self.gui.pause_resume_training_button.setText("Resume")
        elif source.text() == "Resume":
            self.training_state = "continue"
            self.gui.pause_resume_training_button.setText("Pause")
    
    def get_training_speed(self, summary: Summary):
        itps = 1 / (summary.timestamp - self.prev_summary.timestamp) # it/s
        self.prev_summary = summary
        return itps

    def plot_training(self, summary: Summary = None):
        if summary == None:
            return self.training_state
        
        accumulated_step = summary.step + summary.epoch * summary.steps_per_epoch
        self.t.append(accumulated_step)
        self.loss.append(summary.loss)
        self.accuracy.append(summary.training_accuracy)
        if summary.test_accuracy != None:
            self.test_acc.append(summary.test_accuracy)
            self.test_steps.append(accumulated_step)
        self.update_graph_values()

        epoch = summary.epoch + summary.step / summary.steps_per_epoch
        speed = self.get_training_speed(summary)
        del summary

        self.loss_ax.set_title(f"epoch {epoch:.2f} {speed:.2f} it/s")
        self.refresh_graph()

        return self.training_state
    
    def refresh_graph(self):
        self.loss_ax.relim()
        self.loss_ax.autoscale_view()
        self.acc_ax.autoscale_view()

        self.gui.plot_canvas.draw()
        self.gui.plot_canvas.flush_events()
    
    def update_graph_values(self):
        self.loss_line.set_data(self.t, self.loss)
        self.acc_line.set_data(self.t, self.accuracy)
        self.test_acc_line.set_data(self.test_steps, self.test_acc)

    def smooth_test_acc(self):
        # https://www.youtube.com/watch?v=EGsKO9Mye6c
        c = np.c_[self.test_steps, self.test_acc][:,None,:]
        spline = BPoly(c, [0, 1])
        x = np.linspace(0, 1, len(self.t))
        points = spline(x)
        self.test_acc_line.set_data(points[:,0], points[:,1])
        self.refresh_graph()

    def reset_graph(self):
        self.t.clear()
        self.loss.clear()
        self.accuracy.clear()
        self.test_acc.clear()
        self.update_graph_values()

        self.refresh_graph()
        
    def train_ai(self, callback: callable, dataset_options: dict) -> None:
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
        ai = self.models[self.model_to_train]

        dataset = sample.CompiledDataset(
            filename=ai.model.dataset,
            data_augmentation=ai.model.data_augmentation,
            **dataset_options
        )

        steps_per_epoch = (dataset.training_len // ai.model.batch_size) + (dataset.training_len % ai.model.batch_size != 0)
        self.prev_summary = Summary(0, 0, 0, 0, 0, steps_per_epoch, time.time())

        sample.train(
            network=ai.network,
            dataset=dataset,
            callback_training=callback
        )

    #endregion

    #region -x-x-x-x-x-x-x-x-x- Tab : Survey Guess -x-x-x-x-x-x-x-x-x-

    @EventHandler.onclick("next_image_button")
    @EventHandler.onclick("previous_image_button")
    def update_survey_image(self, source: QObject, event: QEvent):
        if source.objectName() == "next_image_button":
            self.next_survey_image(1)
        else:
            self.next_survey_image(-1)
        self.update_survey_guess_field()

    @EventHandler.on(QEvent.KeyPress, keys=[Qt.Key.Key_Return], tab=2)
    def submit_guess(self, source: QObject, event: QEvent):
        guess = self.gui.image_guess_input_line.text()
        if not guess:
            return
        if len(self.get_guesses()) == 200:
            return
        
        img = self.survey_images[self.current_survey_image].filename
        
        percentage = self.save_guess(img, guess)
        self.gui.images_left_progress_bar.setValue(percentage)
        self.next_survey_image(1)
        self.update_survey_guess_field()

    def update_survey_guess_field(self):
        guesses = self.get_guesses()
        img = self.survey_images[self.current_survey_image % 200].filename
        if img in guesses:
            self.gui.image_guess_input_line.setText(guesses[img])
        else:
            self.gui.image_guess_input_line.clear()

    def get_guesses(self):
        name = self.gui.participant_name_line_edit.text()
        guess_file = SURVEY_GUESSES_PATH+f"\\{name}.json"
        if not os.path.exists(guess_file):
            with open(guess_file, "w") as f:
                guesses = {}
                json.dump(guesses, f, indent=4)
        else:
            with open(guess_file, "r") as f:
                guesses = json.load(f)
        return guesses
    
    def save_guess(self, image: str, guess: str):
        name = self.gui.participant_name_line_edit.text()
        guess_file = SURVEY_GUESSES_PATH+f"\\{name}.json"
        guesses = self.get_guesses()
        guesses[image] = guess
        with open(guess_file, "w") as f:
            json.dump(guesses, f, indent=4)
        return 100 * len(guesses) // len(self.survey_images)

    def next_survey_image(self, step: int):
        self.current_survey_image += step
        if self.current_survey_image >= len(self.survey_images):
            return
        self.canvas.set_pixmap(self.gui.guess_canvas_label, 
                              self.survey_images[self.current_survey_image])

    def load_images(self, path: str):
        images = []
        for p in glob.iglob(path+"\\*"):
            pil_img = Image.open(p)
            pil_img.filename = pil_img.filename.split("\\")[-1]
            images.append(pil_img)
        return images
    
    #endregion
    
    #region -x-x-x-x-x-x-x-x-x- Tab : Survey Draw -x-x-x-x-x-x-x-x-x-

    @EventHandler.onclick("choose_model_mappings_button")
    def open_folder_selector(self, source: QObject, event: QEvent):
        f = QtWidgets.QFileDialog.getOpenFileName(self, dir=EMNIST_DATASET_PATH)[0]
        dataset_name = re.search("emnist-[A-z]+\.mat", f).group()
        if dataset_name:
            symbols = list(sample.CompiledDataset(filename=dataset_name).labels)
            symbols *= 5
            random.shuffle(symbols)
            self.gui.symbols_to_draw_list.addItems(symbols)

    @EventHandler.on(QEvent.KeyPress, keys=[Qt.Key.Key_Return], tab=3)
    def save_survey_image(self, source: QObject, event: QEvent):
        if self.gui.symbols_to_draw_list.count() < 1:
            self.show_dialog(QMessageBox.Icon.Warning, "Error", "No mappings selected", 
                             "Select mappings by choosing a dataset")
            return
        
        img = self.canvas.downscaled
        index = 50 - self.gui.symbols_to_draw_list.count() + 50
        symbol_drawn = self.gui.symbols_to_draw_list.takeItem(0).text()
        source = self.gui.image_source_line_edit.text()

        file_name = f"{source}_{symbol_drawn}({index}).png"
        file_path = os.path.join(ANALYSIS_DIGITS_PATH, file_name)
        img.save(file_path, "PNG")
        self.canvas.erease(self.gui.draw_canvas_label)

    #endregion

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Window()
    win.show()
    app.installEventFilter(win)
    sys.exit(app.exec())