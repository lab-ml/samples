import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from labml import tracker, monit, experiment
from labml.configs import option
from labml.helpers.pytorch.datasets.mnist import MNISTConfigs
from labml.helpers.pytorch.device import DeviceConfigs
from labml.helpers.training_loop import TrainingLoopConfigs
from labml.utils import pytorch as pytorch_utils


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Configs(MNISTConfigs, DeviceConfigs, TrainingLoopConfigs):
    epochs: int = 10

    loop_step = 'loop_step'
    loop_count = 'loop_count'

    is_save_models = True

    seed: int = 5
    train_log_interval: int = 10

    is_log_parameters: bool = True

    model: nn.Module

    learning_rate: float = 0.01
    momentum: float = 0.5
    optimizer: optim.SGD

    set_seed = 'set_seed'

    def train(self):
        self.model.train()
        for i, (data, target) in monit.enum("Train", self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            # Add training loss to the logger.
            # The logger will queue the values and output the mean
            tracker.add({'train.loss': loss})
            tracker.add_global_step()

            # Print output to the console
            if i % self.train_log_interval == 0:
                # Output the indicators
                tracker.save()

    def valid(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        idx = 0
        with torch.no_grad():
            for data, target in monit.iterate("Test", self.valid_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = F.nll_loss(output, target, reduction='none')
                values = list(loss.cpu().numpy())
                indexes = [idx + i for i in range(len(values))]
                tracker.add('valid.sample_loss', (indexes, values))

                test_loss += float(np.sum(loss.cpu().numpy()))
                pred = output.argmax(dim=1, keepdim=True)
                values = list(pred.cpu().numpy())
                tracker.add('valid.sample_pred', (indexes, values))
                correct += pred.eq(target.view_as(pred)).sum().item()

                idx += len(values)

        # Add test loss and accuracy to logger
        tracker.add({'valid.loss': test_loss / len(self.valid_dataset)})
        tracker.add({'valid.accuracy': correct / len(self.valid_dataset)})

    def run(self):
        # Training and testing
        tracker.set_queue("train.loss", 20, True)
        tracker.set_histogram("valid.loss", True)
        tracker.set_scalar("valid.accuracy", True)
        tracker.set_indexed_scalar('valid.sample_loss')
        tracker.set_indexed_scalar('valid.sample_pred')

        test_data = np.array([d[0].numpy() for d in self.valid_dataset])
        experiment.save_numpy("valid.data", test_data)

        for _ in self.training_loop:
            self.train()
            self.valid()
            if self.is_log_parameters:
                pytorch_utils.store_model_indicators(self.model)


@option(Configs.model)
def model(c: Configs):
    m: Net = Net()
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


def main():
    conf = Configs()
    experiment.create(name='mnist_indexed_logs', writers={'sqlite'})
    conf.optimizer = 'adam_optimizer'
    experiment.configs(conf,
                                 {},
                                 ['set_seed', 'run'])
    experiment.add_pytorch_models(dict(model=conf.model))
    experiment.start()
    conf.run()


if __name__ == '__main__':
    main()
