import tkinter as tk
from tkinter import ttk

import ai

def goto_button(master, text, page_id):
    return ttk.Button(master, text=text,
            command=lambda: master.master.goto_page(page_id))

class PageMain(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.init_widgets()
        self.place_widgets()

    def init_widgets(self):
        self.btn_train = goto_button(self, "Train Page", "train")
        self.btn_new_model = goto_button(self, "New Model", "new_model")
        self.btn_load_models = ttk.Button(self, text="Load Models", command=ai.load_models)

    def place_widgets(self):
        self.btn_train.pack()
        self.btn_load_models.pack()
        self.btn_new_model.pack()

class PageNewModel(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.init_widgets()
        self.place_widgets()

    def init_widgets(self):
        self.btn_back = goto_button(self, "Back", "main")
        self.btn_done = ttk.Button(self, text="Done", command=self.new_model)
    
    def place_widgets(self):
        self.btn_back.pack()
        self.btn_done.pack()

    def new_model(self):
        self.master.goto_page("main")
    

class PageTrain(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        tk.Label(self, text="Second Page").pack()
        ttk.Button(self, text="Main Page", 
            command=lambda: self.master.goto_page("main")).pack()