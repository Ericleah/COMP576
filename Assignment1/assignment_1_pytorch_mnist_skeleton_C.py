import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import torch.nn.init as init

# Hyperparameters and settings
batch_size = 50
test_batch_size = 1000
epochs = 5
lr = 0.01
try_cuda = True
seed = 1000
logging_interval = 100  # Log every 100 steps
checkpoint_interval = 1100  # Save checkpoint every 1100 steps
logging_dir = 'runs'

# Setting up the logging
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
experiment_name = 'experiment_name'  # Change this for each experiment
log_dir = os.path.join(logging_dir, f'{experiment_name}_{current_time}')
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
        # Define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)
        self.dropout = nn.Dropout(0.5)

        # Initialize weights if using a specific initialization technique
        # Uncomment the following lines if you want to apply Xavier initialization
        # self._initialize_weights()

    def forward(self, x):
        # First Convolutional Layer
        x = self.conv1(x)
        # Choose activation function
        # x_act1 = F.relu(x)
        x_act1 = torch.tanh(x)
        # x_act1 = torch.sigmoid(x)
        # x_act1 = F.leaky_relu(x, negative_slope=0.01)
        x_pool1 = F.max_pool2d(x_act1, 2)

        # Second Convolutional Layer
        x = self.conv2(x_pool1)
        # Choose activation function
        # x_act2 = F.relu(x)
        x_act2 = torch.tanh(x)
        # x_act2 = torch.sigmoid(x)
        # x_act2 = F.leaky_relu(x, negative_slope=0.01)
        x_pool2 = F.max_pool2d(x_act2, 2)

        # Flatten the tensor
        x = x_pool2.view(x.size(0), -1)
        # Fully Connected Layer
        # x_fc1 = F.relu(self.fc1(x))
        x_fc1 = torch.tanh(self.fc1(x))
        # x_fc1 = torch.sigmoid(self.fc1(x))
        # x_fc1 = F.leaky_relu(self.fc1(x), negative_slope=0.01)

        x = self.dropout(x_fc1)
        x_fc2 = self.fc2(x)  # Output layer (logits)

        return x_fc2, x_act1, x_pool1, x_act2, x_pool2, x_fc1

    # Uncomment this method if applying Xavier initialization
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 init.zeros_(m.bias)

# Instantiate the model
model = Net().to(device)

# Apply weight initialization if desired
# Uncomment the following lines to apply Xavier initialization
# def init_weights(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             init.zeros_(m.bias)
# model.apply(init_weights)

# Choose optimizer
# optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adagrad(model.parameters(), lr=lr)

# Loss and Training loops
eps = 1e-13

def log_statistics(layer_name, data, step):
    writer.add_histogram(f'{layer_name}/activations', data, step)
    writer.add_scalar(f'{layer_name}/mean', data.mean().item(), step)
    writer.add_scalar(f'{layer_name}/stddev', data.std().item(), step)
    writer.add_scalar(f'{layer_name}/min', data.min().item(), step)
    writer.add_scalar(f'{layer_name}/max', data.max().item(), step)

def train(epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    step = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, x_act1, x_pool1, x_act2, x_pool2, x_fc1 = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Log statistics every logging_interval steps
        if step % logging_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')
            writer.add_scalar('Loss/train', loss.item(), step)

            # Log histograms and statistics for weights, biases, and activations
            for name, param in model.named_parameters():
                writer.add_histogram(f'{name}/weights', param, step)
                writer.add_scalar(f'{name}/weights_mean', param.mean().item(), step)
                writer.add_scalar(f'{name}/weights_std', param.std().item(), step)
                writer.add_scalar(f'{name}/weights_min', param.min().item(), step)
                writer.add_scalar(f'{name}/weights_max', param.max().item(), step)

            # Log statistics for activations
            log_statistics('Conv1/Activation', x_act1, step)
            log_statistics('Conv1/MaxPool', x_pool1, step)
            log_statistics('Conv2/Activation', x_act2, step)
            log_statistics('Conv2/MaxPool', x_pool2, step)
            log_statistics('FC1/Activations', x_fc1, step)

        step += 1

        # Save checkpoint every checkpoint_interval steps
        if step % checkpoint_interval == 0:
            checkpoint_file = f'./checkpoint_{experiment_name}_{step}.pth'
            torch.save(model.state_dict(), checkpoint_file)

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
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
