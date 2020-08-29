import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

from labml import lab
from labml import monit, tracker, experiment, logger
from labml.configs import BaseConfigs, option
from labml.logger import Text
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


class MNISTLoop:
    def __init__(self, c: 'Configs'):
        self.model = c.model
        self.device = c.device
        self.train_loader = c.train_loader
        self.test_loader = c.test_loader
        self.optimizer = c.optimizer
        self.log_interval = c.log_interval
        self.__epochs = c.epochs
        self.__is_save_models = c.is_save_models
        self.__is_log_parameters = c.is_log_parameters
        self.__log_new_line_interval = c.log_new_line_interval

    def startup(self):
        tracker.set_queue("train.loss", 20, True)
        tracker.set_histogram("valid.loss", True)
        tracker.set_scalar("valid.accuracy", True)

    def _train(self):
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
            if i % self.log_interval == 0:
                # Output the indicators
                tracker.save()

    def _test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in monit.iterate("Test", self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Add test loss and accuracy to logger
        tracker.add({'valid.loss': test_loss / len(self.test_loader.dataset)})
        tracker.add({'valid.accuracy': correct / len(self.test_loader.dataset)})

    def __log_model_params(self):
        if not self.__is_log_parameters:
            return

        # Add histograms with model parameter values and gradients
        pytorch_utils.store_model_indicators(self.model)

    def loop(self):
        # Loop through the monitored iterator
        for epoch in monit.loop(range(0, self.__epochs)):
            self._train()
            self._test()

            self.__log_model_params()

            # Clear line and output to console
            tracker.save()

            # Clear line and go to the next line;
            # that is, we add a new line to the output
            # at the end of each epoch
            if (epoch + 1) % self.__log_new_line_interval == 0:
                logger.log()

            if self.__is_save_models:
                experiment.save_checkpoint()

    def __call__(self):
        self.startup()
        self.loop()


class LoopConfigs(BaseConfigs):
    epochs: int = 10
    is_save_models: bool = True
    is_log_parameters: bool = True
    log_new_line_interval: int = 1


class LoaderConfigs(BaseConfigs):
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader


class Configs(LoopConfigs, LoaderConfigs):
    batch_size: int = 64
    test_batch_size: int = 1000

    # Reset epochs so that it'll be computed
    epochs: int = 10
    use_cuda: bool = True
    cuda_device: int = 0
    seed: int = 5
    log_interval: int = 10

    loop: MNISTLoop

    device: torch.device

    model: nn.Module

    learning_rate: float = 0.01
    momentum: float = 0.5
    optimizer: optim.SGD

    set_seed = None

    not_used: bool = 10


@option(Configs.device)
def device(c: Configs):
    is_cuda = c.use_cuda and torch.cuda.is_available()
    if not is_cuda:
        return torch.device('cpu')
    else:
        if c.cuda_device < torch.cuda.device_count():
            return torch.device('cuda', c.cuda_device)
        else:
            logger.log(f"Cuda device index {c.cuda_device} higher than "
                       f"device count {torch.cuda.device_count()}", Text.warning)
            return torch.device('cuda', torch.cuda.device_count() - 1)


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


def main():
    conf = Configs()
    experiment.create(name='mnist_configs', writers={'sqlite'})
    conf.optimizer = 'sgd_optimizer'
    experiment.configs(conf,
                                 {},
                                 ['set_seed', 'loop'])
    experiment.add_pytorch_models(dict(model=conf.model))
    experiment.start()
    conf.loop()


if __name__ == '__main__':
    main()
