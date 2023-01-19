
import os, sys
sys.path.append(os.getcwd())
from sample import train, CompiledDataset, Network, Model

if __name__ == "__main__":
    import os
    def callback(summary):
        epoch, step, loss, accuracy = summary
        if step % 100 == 0:
            summary = "\n".join([
                f"epoch: {epoch}",
                f"step: {step}",
                f"loss: {loss:.3f}",
                f"accuracy: {accuracy:.3f}",
                # f"data loss: {summary_data[2]:.3f}",
                # f"regularization loss: {summary_data[3]:.3f}",
                # f"learn rate: {summary_data[4]:.6f}"
            ])
            os.system("cls||clear")
            print(summary)

    model = Model("test_model")
    network = Network(model)
    dataset = CompiledDataset(
        filename=model.dataset,
        image_size=(28, 28)
    )
    
    train(
        network=network,
        dataset=dataset,
        callback_training=callback
    )