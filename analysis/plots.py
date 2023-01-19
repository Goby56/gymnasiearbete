import numpy as np
import matplotlib.pyplot as plt
import os

# https://matplotlib.org/stable/tutorials/introductory/pyplot.html

def time(epoch, step):
    return (epoch + 1) * step

v_time = np.vectorize(time)

def accuracy_over_time(epoch, step, accuracy, color: str = "black"):
    X = v_time(epoch, step)
    Y = accuracy

    plt.plot(X, Y, color)
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    #plt.savefig(os.path.join(out_dir, f"accuracy_over_time.{format}"))

def loss_over_time(epoch, step, loss, color: str = "black"):
    X = v_time(epoch, step)
    Y = loss

    plt.plot(X, Y, color)
    plt.xlabel("Time")
    plt.ylabel("Loss")
    #plt.savefig(os.path.join(out_dir, f"loss_over_time.{format}"))

def plt_save(out_dir, name, xlabel=None, ylabel=None, title=None):
    if not xlabel is None:
        plt.xlabel(xlabel)
    if not ylabel is None:
        plt.ylabel(ylabel)
    if not title is None:
        plt.title(title)
    plt.savefig(os.path.join(out_dir, name))
    plt.clf()