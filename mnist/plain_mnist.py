import sys

sys.path.append('../')

import lab

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

import tensorflow as tf


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


class Configs:
    epochs: int = 10

    is_save_models = True
    batch_size: int = 64
    test_batch_size: int = 1000

    use_cuda: bool = True
    cuda_device: int = 0
    seed: int = 5
    train_log_interval: int = 10

    is_log_parameters: bool = True

    device: any

    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader

    summary_writer = tf.summary.create_file_writer('logs/mnist')

    model: nn.Module

    learning_rate: float = 0.01
    momentum: float = 0.5
    optimizer: optim.SGD

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.train_log_interval == 0:
                with self.summary_writer.as_default():
                    tf.summary.scalar('train_loss', loss.item(), step=epoch)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                               len(self.train_loader.dataset),
                                                                               100. * batch_idx / len(
                                                                                   self.train_loader),
                                                                               loss.item()))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        test_accuracy = 100. * correct / len(self.test_loader.dataset)

        with self.summary_writer.as_default():
            tf.summary.scalar('test_loss', test_loss, step=epoch)
            tf.summary.scalar('test_accuracy', test_accuracy, step=epoch)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                     len(self.test_loader.dataset),
                                                                                     test_accuracy))

    def run(self):
        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
            self.test(epoch)

        if self.is_save_models:
            torch.save(self.model.state_dict(), "mnist_cnn.pt")


def cuda(c: Configs):
    is_cuda = c.use_cuda and torch.cuda.is_available()
    if not is_cuda:
        return torch.device("cpu")
    else:
        if c.cuda_device < torch.cuda.device_count():
            return torch.device(f"cuda:{c.cuda_device}")
        else:
            print(f"Cuda device index {c.cuda_device} higher than "  f"device count {torch.cuda.device_count()}")

            return torch.device(f"cuda:{torch.cuda.device_count() - 1}")


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


def data_loaders(c: Configs):
    train = _data_loader(True, c.batch_size)
    test = _data_loader(False, c.test_batch_size)

    return train, test


def model(c: Configs):
    m: Net = Net()
    m.to(c.device)
    return m


def adam_optimizer(c: Configs):
    return optim.Adam(c.model.parameters(), lr=c.learning_rate)


def set_seed(c: Configs):
    torch.manual_seed(c.seed)


def main():
    conf = Configs()

    set_seed(conf)
    conf.device = cuda(conf)

    conf.train_loader, conf.test_loader = data_loaders(conf)
    conf.model = model(conf)
    conf.optimizer = adam_optimizer(conf)

    conf.run()


if __name__ == '__main__':
    main()
