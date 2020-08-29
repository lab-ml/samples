import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from labml import tracker, monit,  experiment, lab
from labml.configs import BaseConfigs, option
from labml.helpers.pytorch.device import DeviceConfigs
from labml.helpers.training_loop import TrainingLoopConfigs
from labml.utils import pytorch as pytorch_utils

from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self, conv1_kernal, conv2_kernal):
        super().__init__()
        self.size = (28 - conv1_kernal - 2 * conv2_kernal + 3) // 4

        self.conv1 = nn.Conv2d(1, 20, conv1_kernal, 1)
        self.conv2 = nn.Conv2d(20, 50, conv2_kernal, 1)
        self.fc1 = nn.Linear(self.size * self.size * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.size * self.size * 50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class LoaderConfigs(BaseConfigs):
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader


class Configs(LoaderConfigs, TrainingLoopConfigs, DeviceConfigs):
    epochs: int = 2  # To test faster

    loop_step = 'loop_step'
    loop_count = 'loop_count'

    is_save_models = True
    batch_size: int = 64
    test_batch_size: int = 1000

    seed: int = 5
    train_log_interval: int = 10

    is_log_parameters: bool = True

    model: nn.Module

    learning_rate: float = 0.01
    momentum: float = 0.5
    optimizer: optim.SGD

    conv1_kernal: int
    conv2_kernal: int

    set_seed = 'set_seed'

    def train(self):
        self.model.train()
        for i, (data, target) in monit.enum("Train", self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

            tracker.add({'train.loss': loss})
            tracker.add_global_step()

            if i % self.train_log_interval == 0:
                tracker.save()

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in monit.iterate("Test", self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target,
                                             reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        tracker.add({'valid.loss': test_loss / len(self.test_loader.dataset)})
        tracker.add({'valid.accuracy': correct / len(self.test_loader.dataset)})

    def run(self):
        tracker.set_queue("train.loss", 20, True)
        tracker.set_histogram("valid.loss", True)
        tracker.set_scalar("valid.accuracy", True)

        for _ in self.training_loop:
            self.train()
            self.test()
            if self.is_log_parameters:
                pytorch_utils.store_model_indicators(self.model)


def _data_loader(is_train, batch_size):
    return torch.utils.data.DataLoader(
        datasets.MNIST(str(lab.get_data_path()),
                       train=is_train,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)


@option([Configs.train_loader, Configs.test_loader])
def data_loaders(c: Configs):
    train = _data_loader(True, c.batch_size)
    test = _data_loader(False, c.test_batch_size)

    return train, test


@option(Configs.model)
def model(c: Configs):
    m: Net = Net(c.conv1_kernal, c.conv2_kernal)
    m.to(c.device)
    return m


@option(Configs.optimizer)
def sgd_optimizer(c: Configs):
    return optim.SGD(c.model.parameters(), lr=c.learning_rate, momentum=c.momentum)


@option(Configs.optimizer)
def adam_optimizer(c: Configs):
    return optim.Adam(c.model.parameters(), lr=c.learning_rate)


@option(Configs.set_seed)
def set_seed(c: Configs):
    torch.manual_seed(c.seed)


@option(Configs.loop_count)
def loop_count(c: Configs):
    return c.epochs * len(c.train_loader)


@option(Configs.loop_step)
def loop_step(c: Configs):
    return len(c.train_loader)


def search(conf: Configs):
    tracker.set_global_step(0)

    experiment.create(name='mnist_hyperparam_tuning')
    experiment.configs(conf,
                                 {},
                                 ['set_seed', 'run'])
    experiment.add_pytorch_models(dict(model=conf.model))
    experiment.start()

    conf.run()
    tracker.reset()


def main():
    for conv1_kernal in [3, 5]:
        for conv2_kernal in [3, 5]:
            conf = Configs()
            conf.conv1_kernal = conv1_kernal
            conf.conv2_kernal = conv2_kernal

            search(conf)


if __name__ == '__main__':
    main()
