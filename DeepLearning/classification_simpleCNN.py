import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from PIL import Image
import scipy
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    # Radom augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])

basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])

# Load MNIST dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=basic_transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.convFirst = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.reluFirst = nn.ReLU()
        self.poolFirst = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        self.convSecond = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.reluSecond = nn.ReLU()
        self.poolSecond = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional layer
        self.convThird = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.reluThird = nn.ReLU()
        self.poolThird = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten layer (no parameters, just reshaping)
        # Full connected layer, 4 = 32 pixel / 2 / 2 / 2since we have three convolutional layers
        self.fcFirst = nn.Linear(128 * 4 * 4, 512)
        self.reluFour = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fcSecond = nn.Linear(512, 10)

    def forward(self, x) :
        # First conv block
        x = self.convFirst(x)
        x = self.reluFirst(x)
        x = self.poolFirst(x)

        # Second conv block
        x = self.convSecond(x)
        x = self.reluSecond(x)
        x = self.poolSecond(x)

        # Third conv block
        x = self.convThird(x)
        x = self.reluThird(x)
        x = self.poolThird(x)

        x = x.flatten(1)

        # Fully connected layer
        x = self.fcFirst(x)
        x = self.reluFour(x)
        x = self.dropout(x)
        x = self.fcSecond(x)
        return x
    
    
# # Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# # Initialize model and move to device 
model = SimpleCNN().to(device)
print(model)

# # Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0005)

def train_epoch(model, train_loader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        # Track progress
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Print every 100 batches
        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_loss = running_loss / 100
            accuracy = 100. * correct / total
            print(f' [{batch_idx * 32}/3200]Loss:{avg_loss:.3f} | Accuracy: {accuracy:.1f}')
            running_loss = 0.0

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return 100. * correct / total
    
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f'\nEpoch: {epoch + 1}')
    train_epoch(model, train_loader, loss_function, optimizer, device)
    accuracy = evaluate(model, test_loader, device)
    print(f'Test Accuracy: {accuracy:.2f}%')



