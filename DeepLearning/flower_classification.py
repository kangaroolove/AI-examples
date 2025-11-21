import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import scipy
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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

train_data = DataLoader(dataset, batch_size=4, shuffle=True)

for images, labels in train_data:
    print(f"Success! Batch shape: {images.shape}")
    break

