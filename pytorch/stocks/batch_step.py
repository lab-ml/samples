from typing import Optional, Callable

import torch
from labml import tracker
from labml.helpers.pytorch.module import Module
from labml.helpers.pytorch.train_valid import BatchStep


class StocksBatchStep(BatchStep):
    def __init__(self, *,
                 model: Module,
                 optimizer: Optional[torch.optim.Adam],
                 loss_func: Callable):
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.model = model

        tracker.set_histogram("*.loss", is_print=True)

        if self.optimizer is None:
            tracker.set_tensor('*.output')

    def log_stats(self, stats: any):
        if self.optimizer is None:
            tracker.add(f'.output', torch.cat(stats['output'], dim=0))

    def prepare_for_iteration(self):
        if self.optimizer is None:
            self.model.eval()
            return True
        else:
            self.model.train()
            return False

    def get_stats(self, data, output):
        stats = {}

        for k in stats.keys():
            stats[k] /= data.shape[1]
        stats['samples'] = len(data)
        if self.optimizer is None:
            stats['output'] = output

        return stats

    def process(self, batch: any):
        device = self.model.device
        # data, target = batch
        data, target = batch['data'], batch['target']
        data, target = data.to(device), target.to(device)
        target = (target - self.model.y_mean) / self.model.y_std

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        output = self.model(data)
        loss = self.loss_func(output, target)

        tracker.add(".loss", loss)

        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()

        output = output * self.model.y_std + self.model.y_mean
        return self.get_stats(data, output)
