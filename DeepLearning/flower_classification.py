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
from torch.utils.data import random_split

#image_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
#labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"

class OxfordFlowersDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'jpg')

        labels_mat = scipy.io.loadmat(os.path.join(root_dir, 'imagelabels.mat'))

        self.labels = labels_mat['labels'][0] - 1
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = f'image_{idx + 1:05d}.jpg'
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label
    
def test_loader(dataset):
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, labels in loader:
        print(f"Success! Batch shape: {images.shape}\n")
        break

class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.transform = transform
        self.indices = indices
        self.dataset = dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = dataset[self.indices[idx]]   # get raw PIL Image / tensor
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    
os.makedirs("flower_data", exist_ok=True)

train_transform = transforms.Compose([
    # Radom augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2),

    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

basic_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = OxfordFlowersDataset("./flower_data", transform=None)
print(f"Total samples: {len(dataset)}")

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

print(f"Training: {len(train_dataset)} images")
print(f"Validation: {len(val_dataset)} images")
print(f"Test: {len(test_dataset)} images")

transformed_train_dataset = TransformSubset(dataset, train_dataset.indices, transform=train_transform)
transformed_valid_dataset = TransformSubset(dataset, val_dataset.indices, transform=basic_transform)
transformed_test_dataset = TransformSubset(dataset, test_dataset.indices, transform=basic_transform)

batch_size = 32
train_loader = DataLoader(transformed_train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(transformed_valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(transformed_test_dataset, batch_size=batch_size, shuffle=False)

class flowerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(3 * 224 * 224, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 102)
        )

    def forward(self, x) :
        x = self.flatten(x)
        x = self.layers(x)
        return x
    
# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# Initialize model and move to device 
model = flowerClassifier().to(device)

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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



