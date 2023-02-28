import os, sys
sys.path.append(os.getcwd())

import numpy as np

from typing import Callable, Optional
import sample
from sample.train import train


class Template:
    def __init__(self,
            model_name: str, 
            dataset_kwargs,
            log_intevall: int = 100
        ):
        print(f"------------ ({model_name}) ------------")

        self.name = model_name
        self.log_intevall = log_intevall
        self.intervall = -1

        self.model = sample.Model(model_name)
        self.network = sample.Network(self.model)
        self.dataset = sample.CompiledDataset(
            filename=self.model.dataset,
            data_augmentation=self.model.data_augmentation,
            #standardize= not any(self.model.data_augmentation.values()),
            **dataset_kwargs
        )
        
        self.training_stats = np.empty((0, 4)) # length of summary data

        self.out_path = os.path.join(os.path.dirname( __file__ ), "results", model_name)
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        print(f"Loaded model...")

    def gather_statistics(self):
        print(f"Starting training process...")
        train(
            network=self.network,
            dataset=self.dataset,
            callback_training=self.callback_training,
            callback_validation=self.callback_validation,
            callback_epoch=self.callback_epoch
        )
        print(f"Training finished!")

    def callback_training(self, summary):
        self.intervall += 1
        if not self.intervall%self.log_intevall:
            self.training_stats = np.r_[self.training_stats, [summary]]
            
    def callback_validation(self, summary):
        pass

    def callback_epoch(self, epoch):
        print(f"Finished epoch {epoch}")
