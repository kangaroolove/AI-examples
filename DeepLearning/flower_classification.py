import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import scipy
import torchvision
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

    
os.makedirs("flower_data", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = OxfordFlowersDataset("./flower_data", transform=transform)
print(f"Total samples: {len(dataset)}")

test_loader(dataset);

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

print(f"Training: {len(train_dataset)} images")
print(f"Validation: {len(val_dataset)} images")
print(f"Test: {len(test_dataset)} images")

batchSize = 32
train_loader = DataLoader(train_dataset, batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batchSize, shuffle=False)



