import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

from labml import lab, tracker, experiment, logger


class Net(nn.Module):
    """
    This is the simple convolutional neural network we use for this sample
    """

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
    """This is the training code"""

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # **✨ Increment the global step**
        tracker.add_global_step()
        # **✨ Store stats in the tracker**
        tracker.save({'loss.train': loss})

        #
        if batch_idx % train_log_interval == 0:
            # **✨ Save added stats**
            tracker.save()


def validate(model, test_loader, device):
    """This is the validation code"""

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

    # **Save stats**
    tracker.save({'loss.valid': test_loss, 'accuracy.valid': test_accuracy})


def main():
    # ✨ Set the types of the stats/indicators.
    # They default to scalars if not specified
    tracker.set_queue('loss.train', 20, True)
    tracker.set_histogram('loss.valid', True)
    tracker.set_scalar('accuracy.valid', True)

    #
    epochs = 10
    train_batch_size = 64
    test_batch_size = 1000
    use_cuda = True
    cuda_device = 0
    seed = 5
    train_log_interval = 10
    learning_rate = 0.01

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

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(str(lab.get_data_path()),
                       train=True,
                       download=True,
                       transform=data_transform),
        batch_size=train_batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST(str(lab.get_data_path()),
                       train=False,
                       download=True,
                       transform=data_transform),
        batch_size=test_batch_size, shuffle=False)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    torch.manual_seed(seed)

    # Create configs object for logging
    configs = {
        'epochs': epochs,
        'train_batch_size': train_batch_size,
        'test_batch_size': test_batch_size,
        'use_cuda': use_cuda,
        'cuda_device': cuda_device,
        'seed': seed,
        'train_log_interval': train_log_interval,
        'learning_rate': learning_rate,
        'device': device,
        'train_loader': train_loader,
        'test_loader': valid_loader,
        'model': model,
        'optimizer': optimizer,
    }

    # ✨ Create the experiment
    experiment.create(name='mnist_labml_tracker')

    # ✨ Save configurations
    experiment.configs(configs)

    # ✨ Set PyTorch models for checkpoint saving and loading
    experiment.add_pytorch_models(dict(model=model))

    # ✨ Start and monitor the experiment
    with experiment.start():
        #
        for epoch in range(1, epochs + 1):
            train(model, optimizer, train_loader, device, train_log_interval)
            validate(model, valid_loader, device)
            logger.log()

    # ✨ Save the models
    experiment.save_checkpoint()

#
if __name__ == '__main__':
    main()
