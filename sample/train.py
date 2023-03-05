from typing import Callable, Optional, NamedTuple
import collections, time

from .network import Network
from .data import CompiledDataset

class Summary(NamedTuple):
    epoch: int
    step: int
    loss: float
    training_accuracy: float
    test_accuracy: float
    steps_per_epoch: int
    timestamp: float

def train(
        network: Network, 
        dataset: CompiledDataset,
        callback_training: Optional[Callable] = None,
        callback_validation: Optional[Callable] = None,
        callback_epoch: Optional[Callable] = None
    ) -> None:
    model = network.model
    assert dataset.shape == model.shape, "Node layout doesn't match with training data."

    training_steps = (dataset.training_len // model.batch_size) + (dataset.training_len % model.batch_size != 0)
    validation_steps = (dataset.validation_len // model.batch_size) + (dataset.validation_len % model.batch_size != 0)

    epoch = 0
    while epoch != model.epochs:

        for step in range(training_steps):
            batch = dataset.next_batch(model.batch_size)
            loss, accuracy = network.train(batch)
            
            if not callback_training is None:
                
                test_accuracy = None
                if step % 10 == 0:
                    test_batch = dataset.next_test_batch(model.batch_size)
                    test_accuracy = network.test(test_batch)
                
                summary = Summary(epoch, step, loss, accuracy, test_accuracy, training_steps, time.time())
                training_state = callback_training(summary)

                if training_state == "continue":
                    continue
                if training_state == "stop":
                    network.save()
                    return
                if training_state == "pause":
                    while True:
                        training_state = callback_training()
                        if training_state == "continue":
                            break
                        if training_state == "stop":
                            network.save()
                            return
                        time.sleep(0.1)

        epoch += 1
    
    network.save()


    # for epoch in range(1, model.epochs+1):
    #     for step in range(training_steps):
    #         batch = dataset.next_batch(model.batch_size)
    #         summary_data = network.train(batch)
            
    #         if not callback_training is None:
    #             training_state = callback_training(summary(epoch, step, *summary_data, training_steps))

    #             if training_state == "continue":
    #                 continue
    #             if training_state == "stop":
    #                 network.save()
    #                 return
    #             if training_state == "pause":
    #                 while True:
    #                     training_state = callback_training(None)
    #                     if training_state == "continue":
    #                         break
    #                     if training_state == "stop":
    #                         network.save()
    #                         return
    #                     time.sleep(0.1)
                    
    #     # for _ in dataset.validation_data:
    #     #     if not callback_validation is None:
    #     #             callback_validation(summary(epoch, step, *summary_data)) # idk wtf is validation

    #     if not callback_epoch is None:
    #         callback_epoch(epoch)

    # network.save()

def stop(network: Network):
    network.save()
