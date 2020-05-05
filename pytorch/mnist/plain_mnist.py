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


def train(epoch, model, optimizer, train_loader, device, train_log_interval, summary_writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % train_log_interval == 0:
            with summary_writer.as_default():
                tf.summary.scalar('train_loss', loss.item(), step=epoch)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(
                                                                               train_loader),
                                                                           loss.item()))


def test(epoch, model, test_loader, device, summary_writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    with summary_writer.as_default():
        tf.summary.scalar('test_loss', test_loss, step=epoch)
        tf.summary.scalar('test_accuracy', test_accuracy, step=epoch)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                 len(test_loader.dataset),
                                                                                 test_accuracy))


def main():
    epochs: int = 10

    is_save_models = True
    train_batch_size: int = 64
    test_batch_size: int = 1000

    use_cuda: bool = True
    cuda_device: int = 0
    seed: int = 5
    train_log_interval: int = 10

    summary_writer = tf.summary.create_file_writer('logs/mnist')

    learning_rate: float = 0.01

    def cuda():
        is_cuda = use_cuda and torch.cuda.is_available()
        if not is_cuda:
            return torch.device("cpu")
        else:
            if cuda_device < torch.cuda.device_count():
                return torch.device(f"cuda:{cuda_device}")
            else:
                print(f"Cuda device index {cuda_device} higher than "  f"device count {torch.cuda.device_count()}")

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

    def data_loaders():
        train_data = _data_loader(True, train_batch_size)
        test_data = _data_loader(False, test_batch_size)

        return train_data, test_data

    def model():
        m: Net = Net()
        m.to(device)
        return m

    def adam_optimizer():
        return optim.Adam(model.parameters(), lr=learning_rate)

    def set_seed():
        torch.manual_seed(seed)

    set_seed()
    device: any = cuda()
    train_loader, test_loader = data_loaders()
    model: nn.Module = model()
    optimizer: optim.adam = adam_optimizer()

    for epoch in range(1, epochs + 1):
        train(epoch, model, optimizer, train_loader, device, train_log_interval, summary_writer)
        test(epoch, model, test_loader, device, summary_writer)

    if is_save_models:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
