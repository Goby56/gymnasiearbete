import tkinter as tk
from tkinter import ttk
import os

import ai

PATH_MODELS = os.path.join(os.getcwd(), "data\\models")

CONFIG_TEMPLATE = """
{
    "setup": {
        "structure": {
            "nodes": [],
            "activations": []
        },
        "normalize_input": false
    },

    "training": {
        "accuracy_function": "",
        "loss_function": "",
        "optimizer": {
            "function": "",
            "args": []
        },
        "optimizer_decay": 0,
        "dataset": "",
        "batch_size": 0,
        "epochs": 0,
        "learn_rate": 0
    }
}
"""

def goto_button(master, text, page_id):
    return ttk.Button(master, text=text,
            command=lambda: master.master.goto_page(page_id))

class PageMain(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        self.models = {}      

        for model_name in os.listdir(PATH_MODELS):
            with open(os.path.join(PATH_MODELS, model_name, "config.json")) as file:
                self.models[model_name] = file.read()

        self.init_widgets()
        self.place_widgets()  

    @property
    def selected_model(self):
        return self.cbx_models.get()

    def init_widgets(self):
        self.txt_config = tk.Text(self, width=60, height=30)

        self.cbx_models = ttk.Combobox(self, values=list(self.models.keys()))
        self.cbx_models.bind("<<ComboboxSelected>>", self.cbx_modified)
        self.btn_save = ttk.Button(self, text="Save", command=self.save_model)
        self.btn_load_nn = ttk.Button(self, text="Create Network", command=self.load_network)

        self.etr_new = ttk.Entry(self, width=20)
        self.btn_new = ttk.Button(self, text="Create New", command=self.new_model)
        self.lbl_new = ttk.Label(self, text="Model Name")

    def place_widgets(self):
        self.lbl_new.grid(row=0, column=0)
        self.etr_new.grid(row=0, column=1)
        self.btn_new.grid(row=0, column=2)

        self.cbx_models.grid(row=2, column=0)
        self.btn_save.grid(row=2, column=1)
        self.btn_load_nn.grid(row=2, column=2)

        self.txt_config.grid(row=1, column=0, columnspan=3)

    def cbx_modified(self, *args, **kwargs):
        self.txt_config.delete("1.0", tk.END)
        self.txt_config.insert(tk.END, self.models[self.selected_model])

    def new_model(self):
        model_name = self.etr_new.get()
        model_path = os.path.join(PATH_MODELS, model_name)
        config_path = os.path.join(model_path, "config.json")
        assert not os.path.exists(model_path)
        os.mkdir(model_path)
        with open(config_path, "w") as _: pass
        self.models[model_name] = CONFIG_TEMPLATE
        self.cbx_models.config(values=list(self.models.keys()))
        self.cbx_models.set(model_name)
        self.cbx_modified()

    def save_model(self):
        self.models[self.selected_model] = self.txt_config.get("1.0", tk.END)
        with open(os.path.join(PATH_MODELS, self.selected_model, "config.json"), "w") as file:
            file.write(self.models[self.selected_model])

    def load_network(self):
        ai.load_model(self.selected_model)
    

class PageTrain(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        tk.Label(self, text="Second Page").pack()
        ttk.Button(self, text="Main Page", 
            command=lambda: self.master.goto_page("main")).pack()