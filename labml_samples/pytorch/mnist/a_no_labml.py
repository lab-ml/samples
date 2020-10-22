import os

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

from labml import lab


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


def train(epoch, model, optimizer, train_loader, device,
          train_log_interval, summary_writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % train_log_interval == 0:
            with summary_writer.as_default():
                tf.summary.scalar('train.loss', loss.item(), step=epoch)

            print(f'train epoch: {epoch}'
                  f' [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]'
                  f'\tLoss: {loss.item():.6f}')


def test(epoch, model, test_loader, device, summary_writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    with summary_writer.as_default():
        tf.summary.scalar('valid.loss', test_loss, step=epoch)
        tf.summary.scalar('valid.accuracy', test_accuracy, step=epoch)

    print(f'\nTest set: Average loss: {test_loss:.4f},'
          f' Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({test_accuracy:.0f}%)\n')


def main():
    epochs = 10

    is_save_models = True
    train_batch_size = 64
    test_batch_size = 1000

    use_cuda = True
    seed = 5
    train_log_interval = 10

    learning_rate = 0.01

    # get device
    is_cuda = use_cuda and torch.cuda.is_available()
    if not is_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:0")

    # data transform
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # train loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(str(lab.get_data_path()),
                       train=True,
                       download=True,
                       transform=data_transform),
        batch_size=train_batch_size, shuffle=True)

    # test loader
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(str(lab.get_data_path()),
                       train=False,
                       download=True,
                       transform=data_transform),
        batch_size=test_batch_size, shuffle=False)

    # model
    model = Net().to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # set seeds
    torch.manual_seed(seed)

    # tensorboard writer
    summary_writer = tf.summary.create_file_writer(os.path.join(os.getcwd(), 'logs/mnist'))

    # training loop
    for epoch in range(1, epochs + 1):
        train(epoch, model, optimizer, train_loader, device,
              train_log_interval, summary_writer)
        test(epoch, model, test_loader, device, summary_writer)

    if is_save_models:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
