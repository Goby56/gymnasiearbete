import numpy as np
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw

import os, sys
sys.path.append(os.getcwd())
import sample

PATH_MODELS = os.path.join(os.getcwd(), "data\\models")

class Canvas(tk.Frame):
    def __init__(self, master, callback=None, size: int=256, target_size:tuple[int, int]=(28, 28)):
        super().__init__(master)
        self.callback_ = callback
        self.target_size = target_size
        self.size = size

        self.canvas = tk.Canvas(self, background="black", width=size, height=size)
        self.brush_scale = ttk.Scale(self, from_=1, to=10, command=self.change_brush)
        self.label_var = tk.StringVar(value="Brush Size   (1)")
        self.scale_label = tk.Label(self, textvariable=self.label_var)

        self.canvas.bind('<1>', self.activate_paint)
        self.canvas.bind('<3>', self.clear)
        self.canvas.bind("<ButtonRelease-1>", self.callback)

        self.canvas.grid(row=0, column=0, columnspan=2, sticky="news")
        self.scale_label.grid(row=1, column=0)
        self.brush_scale.grid(row=1, column=1)

        #PIL
        self.image = Image.new("L", (size, size))
        self.draw = ImageDraw.Draw(self.image)
        
        self.brush_size = 1
        self.lastx, self.lasty = None, None
        
    def clear(self, e):
        self.canvas.delete("all")
        #PIL
        self.draw.rectangle((0, 0, self.size, self.size), fill="black")

    def callback(self, e):
        if not self.callback_ is None:
            image = self.image.resize(self.target_size)
            self.callback_(np.asarray(image))

    def change_brush(self, e):
        self.brush_size = self.brush_scale.get()
        self.label_var.set(f"Brush Size   ({self.brush_size:.0f})")

    def activate_paint(self, e):
        self.canvas.bind('<B1-Motion>', self.paint)
        self.lastx, self.lasty = e.x, e.y

    def paint(self, e):
        x, y = e.x, e.y
        self.canvas.create_line((self.lastx, self.lasty, x, y), width=self.brush_size, fill="white")
        #PIL
        self.draw.line((self.lastx, self.lasty, x, y), width=int(self.brush_size), fill="white")
        self.lastx, self.lasty = x, y

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Playground")
        self.resizable(False, False)
        self.canvas = Canvas(self, self.callback)

        right_panel = tk.Frame(self, background="black")
        self.model_selector = ttk.Combobox(right_panel)
        self.model_selector.bind("<<ComboboxSelected>>", self.on_select)
        self.guesses = tk.Listbox(right_panel)

        self.model_selector.pack(fill="x")
        self.guesses.pack(fill="both", expand=True)

        self.canvas.grid(row=0, column=0)
        right_panel.grid(row=0, column=1, sticky="news")
        
        self.models = {}
        self.load_models()
        self.last_callback = None

    @property
    def current_model(self):
        return self.model_selector.get()

    def on_select(self, e):
        if not self.last_callback is None:
            self.callback(self.last_callback)
    
    def load_models(self):
        models = os.listdir(PATH_MODELS)
        for model_name in models:
            model = sample.Model(model_name)
            self.models[model_name] = (sample.Network(model), model)
        self.model_selector.config(values=models, state="readonly")
        self.model_selector.current(0)

    def callback(self, array):
        self.last_callback = array
        input_vector = np.c_[array.flatten()/255].T
        output_vector = self.models[self.current_model][0].forward(input_vector).flatten()
        labels = self.models[self.current_model][1].mapping
        guesses = list(zip(labels, output_vector))
        guesses.sort(key=lambda x: x[1])
        self.guesses.delete(0, tk.END)
        for guess in guesses:
            string = f"{guess[0]}: ({100*guess[1]:.1f}%)"
            self.guesses.insert(0, string)

print()
app = Application()
app.mainloop()