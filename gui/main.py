import sys, cv2, os, random, glob, json, re, collections, matplotlib, time
import threading
import numpy as np

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
SURVEY_GUESSES_PATH = os.path.join(os.getcwd(), "survey\\guesses")
MODELS_PATH = os.path.join(os.getcwd(), "data\\models")
TRAINING_GRAPHS_PATH = os.path.join(os.getcwd(), "analysis\\results")

SYMBOL_MAPPINGS = "AaBbCDdEeFfGgHhIJKkLMNnOPQqRrSTtUVWXYZ0123456789"

import sample

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
            # if type(source) == type(ref):
            #     continue
            if type(ref) == SurveyDialog:
                print(callback)
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
        self.load_models(blacklist=[])

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
        self.accuracy = []
        self.loss = []
        
        self.loss_ax = self.graph.add_subplot(111)
        self.loss_ax.set_ylabel("loss")
        self.loss_ax.set_xlabel("steps")
        self.loss_line, = self.loss_ax.plot(self.t, self.loss, label="loss", color="blue")
        self.loss_ax.set_ylim([0,5])

        self.acc_ax = self.loss_ax.twinx()
        self.acc_ax.set_ylabel("accuracy")
        self.acc_line, = self.acc_ax.plot(self.t, self.accuracy, label="accuracy", color="orange")
        self.acc_ax.set_ylim([0,1])

        lines = [self.loss_line, self.acc_line]
        labels = [l.get_label() for l in lines]
        self.loss_ax.legend(lines, labels, loc="center right")

        # SURVEY STUFF
        self.survey_participant = "karl"
        self.survey_images = self.load_images(SURVEY_IMAGES_PATH)
        random.seed(69)
        random.shuffle(self.survey_images)
        self.current_survey_image = 0
        self.canvas.set_pixmap(self.gui.guess_canvas_label, 
                              self.survey_images[self.current_survey_image])
        self.gui.images_left_progress_bar.setValue(0)

        # ADD SURVEY SAMPLES
        self.gui.symbols_to_draw_list.addItems(random.sample(SYMBOL_MAPPINGS, 25))

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

    @EventHandler.onclick("toggle_graph_toolbar")
    def toggle_graph_toolbar(self, source: QObject, event: QEvent):
        if self.gui.plot_toolbar.isHidden():
            self.gui.plot_toolbar.show()
            source.setText("▲")
        else:
            self.gui.plot_toolbar.hide()
            source.setText("▼")

    @EventHandler.onclick("configure_training_button")
    def select_training_config(self, source: QObject, event: QEvent):
        f = QtWidgets.QFileDialog.getOpenFileName(self)[0]
        model_name = re.search("[A-z]+(?=\/config\.json)", f)
        if model_name:
            self.model_to_train = model_name.group()

    @EventHandler.onclick("start_stop_training_button")
    def handle_training(self, source: QObject, event: QEvent):
        if not self.model_to_train:
            self.show_dialog(QMessageBox.Icon.Warning, "Error", "No module selected")
            return
        if source.text() == "Start":
            self.training_state = "continue"
            options = {
                "image_size": (28, 28),
                "subtract_label": True
            }
            self.training_thread = threading.Thread(target=self.train_ai, args=(self.plot_training, options))
            self.training_thread.start()
            source.setText("Stop")
            self.gui.pause_resume_training_button.show()
        elif source.text() == "Stop":
            self.training_state = "stop"
            self.gui.pause_resume_training_button.hide()
            self.gui.pause_resume_training_button.setText("Pause")
            
            sess_count = self.session.count + 1
            epoch, excess = self.get_epoch_progress()
            fname = f"sess{sess_count}-ep{epoch}-{excess}%-({self.session.steps_per_epoch}).png"
            path = TRAINING_GRAPHS_PATH+f"\\{self.model_to_train}\\{fname}"
            self.graph.savefig(path)
            self.reset_graph()
            source.setText("Start")

    @EventHandler.on(QEvent.Close)
    def stop_training(self, source: QObject, event: QEvent):
        self.training_state = "stop"

    def get_epoch_progress(self):
        epoch = self.session.epoch + self.training_summary.epoch - 1
        excess = self.session.excess + int(100 * (self.training_summary.step+1) / self.session.steps_per_epoch)
        epoch += excess // 100
        excess %= 100
        return epoch, excess

    @EventHandler.onclick("pause_resume_training_button")
    def toggle_training(self, source: QObject, event: QEvent):
        if source.text() == "Pause":
            self.training_state = "pause"
            self.gui.pause_resume_training_button.setText("Resume")
        elif source.text() == "Resume":
            self.training_state = "continue"
            self.gui.pause_resume_training_button.setText("Pause")

    def get_previous_session(self, model_name: str, steps_per_epoch: int = None):
        model_path = TRAINING_GRAPHS_PATH+f"\\{model_name}"
        sessions = list(filter(lambda fname: re.search("sess[0-9]+-ep[0-9]+-[0-9]{1,2}%-\([0-9]+\)\.png", fname), 
                          os.listdir(model_path)))
        if len(sessions) < 1:
            return Session(0, 0, 0, steps_per_epoch)
        sessions.sort(key=(lambda s: int(re.search("(?<=sess)[0-9]+", s).group())))
        prev_sess = sessions[-1]
        count = len(sessions)
        epoch = int(re.search("(?<=ep)[0-9]+", prev_sess).group())
        excess = int(re.search("[0-9]+(?=%)", prev_sess).group())
        steps_per_epoch = int(re.search("(?<=\()[0-9]+(?=\))", prev_sess).group())
        return Session(count, epoch, excess, steps_per_epoch)

    def plot_training(self, summary: Union[collections.namedtuple, None]):
        if summary == None:
            return self.training_state
        
        self.t.append(summary.step)
        self.loss.append(summary.loss)
        self.accuracy.append(summary.accuracy)
        self.update_graph_values()

        self.training_summary = summary

        epoch, excess = self.get_epoch_progress()

        self.loss_ax.set_title(f"epoch {epoch+excess/100:.2f}")
        self.refresh_graph()

        self.training_summary = summary

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

    def reset_graph(self):
        self.t.clear()
        self.loss.clear()
        self.accuracy.clear()
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
        assert "image_size" in dataset_options, "image_size must be included in dataset_options"

        dataset = sample.CompiledDataset(
            filename=ai.model.dataset,
            data_augmentation=ai.model.data_augmentation,
            **dataset_options
        )

        steps_per_epoch = (dataset.training_len // ai.model.batch_size) + (dataset.training_len % ai.model.batch_size != 0)
        self.session = self.get_previous_session(self.model_to_train, steps_per_epoch)

        sample.train(
            network=ai.network,
            dataset=dataset,
            callback_training=callback
        )

    #endregion

    #region -x-x-x-x-x-x-x-x-x- Tab : Survey Guess -x-x-x-x-x-x-x-x-x-

    @EventHandler.onclick("survey_config_button")
    def open_survey_dialog(self, source: QObject, event: QEvent):
        def on_submit(name, path):
            print(path)
            # for p in glob.iglob(path+"*")
            
        self.survey_dialog = SurveyDialog(on_submit)
        self.survey_dialog.show()

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
        img = self.survey_images[self.current_survey_image].filename
        
        percentage = self.save_guess(img, guess)
        self.gui.images_left_progress_bar.setValue(percentage)
        self.next_survey_image(1)
        self.update_survey_guess_field()

    def update_survey_guess_field(self):
        guesses = self.get_guesses()
        img = self.survey_images[self.current_survey_image].filename
        if img in guesses:
            self.gui.image_guess_input_line.setText(guesses[img])
        else:
            self.gui.image_guess_input_line.clear()

    def get_guesses(self):
        guess_file = SURVEY_GUESSES_PATH+f"\\{self.survey_participant}.json"
        if not os.path.exists(guess_file):
            with open(guess_file, "w") as f:
                guesses = {}
                json.dump(guesses, f, indent=4)
        else:
            with open(guess_file, "r") as f:
                guesses = json.load(f)
        return guesses
    
    def save_guess(self, image: str, guess: str):
        guess_file = SURVEY_GUESSES_PATH+f"\\{self.survey_participant}.json"
        guesses = self.get_guesses()
        guesses[image] = guess
        with open(guess_file, "w") as f:
            json.dump(guesses, f, indent=4)
        return 100 * len(guesses) // len(self.survey_images)

    def next_survey_image(self, step: int):
        self.current_survey_image += step
        if self.current_survey_image > len(self.survey_images):
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

    @staticmethod
    def save_emnist_images(amount: int):
        """
        Only used to load and save emnist images to be used in the survey.
        """
        dataset = sample.CompiledDataset(filename="emnist-balanced.mat", image_size=(28, 28))
        samples = dataset.get(amount, convert=True)
        occurences = {}
        for i in range(amount):
            label = samples[i][1]
            if label in occurences:
                occurences[label] += 1
            else:
                occurences[label] = 0            
            pil_img = Image.fromarray(samples[i][0]).convert("RGB")
            fn = f"emnist_{samples[i][1]}({occurences[label]}).png"
            path = SURVEY_IMAGES_PATH+f"\\{fn}"
            pil_img.save(path)
         
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
        self.installEventFilter(self)
        self.setModal(True)
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
        print(self)
        self.participant_name = self.gui.participant_name_line_edit.text()
        if self.folder_selection and self.participant_name:
            self.on_submit(self.participant_name, self.folder_selection)
        else:
            # Raise error if not enough information was provided
            pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Window()
    win.show()
    app.installEventFilter(win)
    sys.exit(app.exec())