import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from labml import experiment, lab
from labml.configs import BaseConfigs

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


@LoaderConfigs.calc([LoaderConfigs.train_loader, LoaderConfigs.test_loader])
def data_loaders(c: LoaderConfigs):
    train_data = _data_loader(True, c.train_batch_size)
    test_data = _data_loader(False, c.test_batch_size)

    return train_data, test_data


class Configs(LoaderConfigs):
    epochs: int = 10

    use_cuda: bool = True
    cuda_device: int = 0
    seed: int = 5

    device: any

    model: nn.Module

    learning_rate: float = 0.01
    momentum: float = 0.5
    optimizer: optim.SGD

    set_seed = 'set_seed'

    def train(self, epoch):
        self.model.train()
        for i, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

            print(f'train epoch: {epoch}'
                  f' [{i * len(data)}/{len(self.train_loader.dataset)}'
                  f' ({100. * i / len(self.train_loader):.0f}%)]'
                  f'\tloss: {loss.item():.6f}')

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += F.cross_entropy(output, target,
                                             reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        test_accuracy = 100. * correct / len(self.test_loader.dataset)

        print(f'\ntest set: average loss: {test_loss:.4f},'
              f' accuracy: {correct}/{len(self.test_loader.dataset)}'
              f' ({test_accuracy:.0f}%)\n')

    def run(self):
        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
            self.test()


@Configs.calc(Configs.device)
def get_device(c: Configs):
    is_cuda = c.use_cuda and torch.cuda.is_available()
    if not is_cuda:
        return torch.device("cpu")
    else:
        if c.cuda_device < torch.cuda.device_count():
            return torch.device(f"cuda:{c.cuda_device}")
        else:
            print(f"cuda device index {c.cuda_device} higher than "
                  f"device count {torch.cuda.device_count()}")

            return torch.device(f"cuda:{torch.cuda.device_count() - 1}")


@Configs.calc(Configs.model)
def model(c: Configs):
    m: Net = Net()
    m.to(c.device)
    return m


@Configs.calc(Configs.optimizer)
def sgd_optimizer(c: Configs):
    return optim.SGD(c.model.parameters(), lr=c.learning_rate, momentum=c.momentum)


@Configs.calc(Configs.optimizer)
def adam_optimizer(c: Configs):
    return optim.Adam(c.model.parameters(), lr=c.learning_rate)


@Configs.calc(Configs.set_seed)
def set_seed(c: Configs):
    torch.manual_seed(c.seed)


def main():
    conf = Configs()
    conf.optimizer = 'adam_optimizer'
    experiment.create(name='configs')
    experiment.calculate_configs(conf,
                                 {},
                                 ['set_seed', 'run'])
    experiment.start()
    conf.run()


if __name__ == '__main__':
    main()
