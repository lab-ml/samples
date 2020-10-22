import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from labml import experiment, lab, tracker
from labml.configs import BaseConfigs, option

from torchvision import datasets, transforms


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
        return self.fc2(x)


class LoaderConfigs(BaseConfigs):
    train_batch_size: int = 64
    test_batch_size: int = 1000

    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader


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


@option([LoaderConfigs.train_loader, LoaderConfigs.test_loader])
def data_loaders(c: LoaderConfigs):
    train_data = _data_loader(True, c.train_batch_size)
    test_data = _data_loader(False, c.test_batch_size)

    return train_data, test_data


class Configs(LoaderConfigs):
    epochs: int = 10

    use_cuda: bool = True
    seed: int = 5

    device: any

    model: nn.Module

    learning_rate: float = 0.01
    momentum: float = 0.5
    optimizer: optim.Adam = 'adam_optimizer'

    set_seed: int

    train_log_interval = 10

    def train(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

            # **✨ Increment the global step**
            tracker.add_global_step()
            # **✨ Store stats in the tracker**
            tracker.save({'loss.train': loss})

            #
            if batch_idx % self.train_log_interval == 0:
                # **✨ Save added stats**
                tracker.save()

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        test_accuracy = 100. * correct / len(self.test_loader.dataset)

        # **Save stats**
        tracker.save({'loss.valid': test_loss, 'accuracy.valid': test_accuracy})

    def run(self):
        for epoch in range(1, self.epochs + 1):
            self.train()
            self.test()


@option(Configs.device)
def get_device(c: Configs):
    is_cuda = c.use_cuda and torch.cuda.is_available()
    if not is_cuda:
        return torch.device("cpu")
    else:
        return torch.device(f"cuda")


@option(Configs.model)
def model(c: Configs):
    return Net().to(c.device)


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
    experiment.create(name='configs')
    experiment.configs(conf,
                       {'optimizer': 'sgd_optimizer'},
                       ['set_seed', 'run'])
    experiment.start()
    conf.run()

    # save the model
    experiment.save_checkpoint()


if __name__ == '__main__':
    main()
