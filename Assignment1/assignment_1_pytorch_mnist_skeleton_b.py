import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path

batch_size = 50
test_batch_size = 1000
epochs = 5
lr = 0.01
try_cuda = True
seed = 1000
logging_interval = 100  # Log every 100 steps
checkpoint_interval = 1100  # Save checkpoint every 1100 steps
logging_dir = 'runs'

# 1) setting up the logging
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join(logging_dir, current_time)
writer = SummaryWriter(log_dir=log_dir)

# Deciding whether to use GPU
if torch.cuda.is_available() and try_cuda:
    device = torch.device('cuda')
    torch.cuda.manual_seed(seed)
else:
    device = torch.device('cpu')
    torch.manual_seed(seed)

# Data Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transform),
    batch_size=test_batch_size, shuffle=False
)

# Network Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x_relu1 = F.relu(x)
        x_pool1 = F.max_pool2d(x_relu1, 2)

        x = self.conv2(x_pool1)
        x_relu2 = F.relu(x)
        x_pool2 = F.max_pool2d(x_relu2, 2)

        x = x_pool2.view(x.size(0), -1)
        x_fc1 = F.relu(self.fc1(x))
        x = self.dropout(x_fc1)
        x_fc2 = self.fc2(x)

        return x_fc2, x_relu1, x_pool1, x_relu2, x_pool2, x_fc1


model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Loss and Training loops
eps = 1e-13

def log_statistics(layer_name, data, step):
    writer.add_histogram(f'{layer_name}/activations', data, step)
    writer.add_scalar(f'{layer_name}/mean', data.mean(), step)
    writer.add_scalar(f'{layer_name}/stddev', data.std(), step)
    writer.add_scalar(f'{layer_name}/min', data.min(), step)
    writer.add_scalar(f'{layer_name}/max', data.max(), step)

def train(epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    step = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, x_relu1, x_pool1, x_relu2, x_pool2, x_fc1 = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Log statistics every 100 steps
        if step % logging_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')
            writer.add_scalar('Loss/train', loss.item(), step)

            # Log histograms and statistics for weights, biases, and activations
            for name, param in model.named_parameters():
                writer.add_histogram(f'{name}/weights', param, step)
                writer.add_scalar(f'{name}/weights_mean', param.mean(), step)
                writer.add_scalar(f'{name}/weights_std', param.std(), step)

            # Log statistics for activations
            log_statistics('Conv1/Relu', x_relu1, step)
            log_statistics('Conv1/MaxPool', x_pool1, step)
            log_statistics('Conv2/Relu', x_relu2, step)
            log_statistics('Conv2/MaxPool', x_pool2, step)
            log_statistics('FC1/Activations', x_fc1, step)

        step += 1

        # Save checkpoint every 1100 steps
        if step % checkpoint_interval == 0:
            checkpoint_file = f'./checkpoint_{step}.pth'
            torch.save(model.state_dict(), checkpoint_file)

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output, _, _, _, _, _ = model(data)

        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.2f}%)\n')

    # Log test loss and accuracy to TensorBoard
    writer.add_scalar('Test/Loss', test_loss, epoch)
    writer.add_scalar('Test/Accuracy', test_accuracy, epoch)

# Run training and testing loops
for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)

writer.close()

# Uncomment the following lines to run TensorBoard
# %load_ext tensorboard
# %tensorboard --logdir=runs
