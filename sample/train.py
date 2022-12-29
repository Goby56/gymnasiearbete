import data, os
from network import Network, Mode
from model import Model

def __train():
    model = Model("test_model")
    network = Network(model, mode=Mode.train)
    dataset = data.CompiledDataset(
        filename=model.dataset, 
        validation_partition=True, 
        as_array=True, 
        flatten=True, 
        normalize=True
    )

    assert dataset.shape == model.shape

    show_summary_every = 3 # steps

    training_steps = (dataset.training_len // model.batch_size) + (dataset.training_len % model.batch_size != 0)
    validation_steps = (dataset.validation_len // model.batch_size) + (dataset.validation_len % model.batch_size != 0)

    for epoch in range(1, model.epochs+1):
        for step in range(training_steps):
            summary_data = network.train(dataset.next_batch(model.batch_size))
            if step % show_summary_every == 0:
                summary = "\n".join([
                    f"epoch: {epoch}",
                    f"step: {step}",
                    f"accuracy: {summary_data[0]:.3f}",
                    f"loss: {summary_data[1]:.3f}",
                    f"data loss: {summary_data[2]:.3f}",
                    f"regularization loss: {summary_data[3]:.3f}",
                    f"learn rate: {summary_data[4]:.6f}"
                ])
                if network.mode == Mode.train:
                    os.system("cls||clear")
                    print(summary)

if __name__ == "__main__":
    __train()




    