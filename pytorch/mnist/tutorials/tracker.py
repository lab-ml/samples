import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from labml import lab, tracker, experiment, logger
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


def train(model, optimizer, train_loader, device, train_log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        tracker.add({'train.loss': loss})
        tracker.add_global_step()

        if batch_idx % train_log_interval == 0:
            tracker.save()


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    tracker.add({'valid.loss': test_loss})
    tracker.add({'valid.accuracy': test_accuracy})
    tracker.save()


def main():
    epochs = 10

    train_batch_size = 64
    test_batch_size = 1000

    use_cuda = True
    cuda_device = 0
    seed = 5
    train_log_interval = 10

    learning_rate = 0.01

    # get device
    is_cuda = use_cuda and torch.cuda.is_available()
    if not is_cuda:
        device = torch.device("cpu")
    else:
        if cuda_device < torch.cuda.device_count():
            device = torch.device(f"cuda:{cuda_device}")
        else:
            print(f"Cuda device index {cuda_device} higher than "
                  f"device count {torch.cuda.device_count()}")

            device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")

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

    # set metrics types
    tracker.set_queue("train.loss", 20, True)
    tracker.set_histogram("valid.loss", True)
    tracker.set_scalar("valid.accuracy", True)

    # create the experiment
    experiment.create(name='tracker')
    experiment.add_pytorch_models(dict(model=model))
    experiment.start()

    # training loop
    for epoch in range(1, epochs + 1):
        train(model, optimizer, train_loader, device, train_log_interval)
        test(model, test_loader, device)
        logger.log()

    # save the model
    experiment.save_checkpoint()


if __name__ == '__main__':
    main()
