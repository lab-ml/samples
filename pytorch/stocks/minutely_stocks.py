from typing import Tuple, List

import torch
from labml import experiment, tracker
from labml.configs import option, calculate
from labml.helpers.pytorch.device import DeviceConfigs
from labml.helpers.pytorch.train_valid import TrainValidConfigs
from torch import nn
from torch.utils.data import DataLoader

from pytorch.stocks.batch_step import StocksBatchStep
from pytorch.stocks.dataset import MinutelyDataset
from pytorch.stocks.model import CnnModel


class Configs(DeviceConfigs, TrainValidConfigs):
    epochs = 1000
    dropout: float
    validation_dates: int = 100
    train_dataset: MinutelyDataset
    valid_dataset: MinutelyDataset
    model: CnnModel
    conv_sizes: List[Tuple[int, int]]
    activation: nn.Module
    accuracy_func = None
    train_batch_step = 'train_stocks_batch_step'
    valid_batch_step = 'valid_stocks_batch_step'
    learning_rate: float = 1e-4
    train_batch_size: int = 32
    valid_batch_size: int = 64


@option(Configs.train_loader)
def train_loader(c: Configs):
    return DataLoader(c.train_dataset,
                      batch_size=c.train_batch_size,
                      shuffle=True)


@option(Configs.valid_loader)
def train_loader(c: Configs):
    return DataLoader(c.valid_dataset,
                      batch_size=c.valid_batch_size,
                      shuffle=False)


@option(Configs.train_dataset)
def train_dataset(c: Configs):
    return MinutelyDataset(-c.validation_dates)


@option(Configs.valid_dataset)
def train_dataset(c: Configs):
    return MinutelyDataset(c.validation_dates)


@option(Configs.model)
def cnn_model(c: Configs):
    return CnnModel(price_mean=c.train_dataset.price_mean,
                    price_std=c.train_dataset.price_std,
                    volume_mean=c.train_dataset.volume_mean,
                    volume_std=c.train_dataset.volume_std,
                    y_mean=c.train_dataset.y_mean,
                    y_std=c.train_dataset.y_std,
                    activation=c.activation,
                    conv_sizes=c.conv_sizes,
                    dropout=c.dropout).to(c.device)


calculate(Configs.activation, 'relu', [], lambda: nn.ReLU())
calculate(Configs.activation, 'sigmoid', [], lambda: nn.Sigmoid())


@option(Configs.optimizer)
def adam_optimizer(c: Configs):
    return torch.optim.Adam(c.model.parameters(), lr=c.learning_rate)


@option(Configs.loss_func)
def loss_func():
    return nn.MSELoss()


@option(TrainValidConfigs.train_batch_step)
def train_stocks_batch_step(c: TrainValidConfigs):
    return StocksBatchStep(model=c.model,
                           optimizer=c.optimizer,
                           loss_func=c.loss_func)


@option(TrainValidConfigs.valid_batch_step)
def valid_stocks_batch_step(c: TrainValidConfigs):
    return StocksBatchStep(model=c.model,
                           optimizer=None,
                           loss_func=c.loss_func)


def main():
    experiment.create()
    conf = Configs()
    conf.learning_rate = 1e-4
    conf.epochs = 500
    conf.conv_sizes = [(128, 2), (256, 4)]
    # conf.conv_sizes = [(128, 1), (256, 2)]
    conf.activation = 'relu'
    conf.dropout = 0.1
    conf.train_batch_size = 32
    experiment.configs(conf, 'run')

    experiment.start()
    with tracker.namespace('valid'):
        conf.valid_dataset.save_artifacts()
    conf.run()


if __name__ == '__main__':
    main()
