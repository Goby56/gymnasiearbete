import tkinter as tk

import pages

class Application(tk.Frame):
    def __init__(self, master):
        super().__init__(master)

        self.pages = {
            "main": pages.PageMain(self),
            "train": pages.PageTrain(self),
            "new_model": pages.PageNewModel(self)
        }
        self.current_page = self.pages["main"]
        self.current_page.pack()

    def goto_page(self, page: str):
        self.current_page.pack_forget()
        self.current_page = self.pages[page]
        self.current_page.pack()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = Application(root)
    app.pack()
    root.mainloop()

