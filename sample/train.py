import os, sys
import numpy as np

try:
    from .network import Network
    from .model import Model
    from .data import CompiledDataset
except ImportError:
    from network import Network
    from model import Model
    from data_old import CompiledDataset

def train():
    model = Model("test_new_data")
    network = Network(model)
    dataset = CompiledDataset(
        filename=model.dataset,
        validation_partition=True, 
        as_array=True, 
        flatten=True,
        normalize=True
    )
    assert dataset.shape == model.shape

    show_summary_every = 100 # steps

    training_steps = (dataset.training_len // model.batch_size) + (dataset.training_len % model.batch_size != 0)
    validation_steps = (dataset.validation_len // model.batch_size) + (dataset.validation_len % model.batch_size != 0)

    for epoch in range(1, model.epochs+1):
        #dataset.generate_training_data()
        for step in range(training_steps):
            batch = dataset.next_batch(model.batch_size)
            if batch == None:
                break
            summary_data = network.train(batch)
            if step % show_summary_every == 0:
                summary = "\n".join([
                    f"epoch: {epoch}",
                    f"step: {step}",
                    f"accuracy: {summary_data[1]:.3f}",
                    f"loss: {summary_data[0]:.3f}",
                    # f"data loss: {summary_data[2]:.3f}",
                    # f"regularization loss: {summary_data[3]:.3f}",
                    # f"learn rate: {summary_data[4]:.6f}"
                ])
                os.system("cls||clear")
                print(summary)
    network.save()

if __name__ == "__main__":
    train()




    