from typing import Callable, Optional
import collections

from .network import Network
from .data import CompiledDataset

def train(
        network: Network, 
        dataset: CompiledDataset,
        callback_training: Optional[Callable] = None,
        callback_validation: Optional[Callable] = None,
        callback_epoch: Optional[Callable] = None
    ) -> None:
    model = network.model
    assert dataset.shape == model.shape

    training_steps = (dataset.training_len // model.batch_size) + (dataset.training_len % model.batch_size != 0)
    validation_steps = (dataset.validation_len // model.batch_size) + (dataset.validation_len % model.batch_size != 0)
    summary = collections.namedtuple("summary", ["epoch", "step", "loss", "accuracy"])

    for epoch in range(1, model.epochs+1):
        for step in range(training_steps):
            batch = dataset.next_batch(model.batch_size)
            summary_data = network.train(batch)
            if not callback_training is None:
                callback_training(summary(epoch, step, *summary_data))
        # for _ in dataset.validation_data:
        #     if not callback_validation is None:
        #             callback_validation(summary(epoch, step, *summary_data)) # idk wtf is validation
        if not callback_epoch is None:
            callback_epoch(epoch)

    network.save()
