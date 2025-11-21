import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import scipy
import urllib.request

#image_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
#labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"

class OxfordFlowersDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'jpg')

        labels_mat = scipy.io.loadmat(os.path.join(root_dir, 'imagelabels.mat'))

        self.labels = labels_mat['labels'][0] - 1

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = f'image_{idx + 1:05d}.jpg'
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path)
        label = self.labels[idx]

        return image, label
    
os.makedirs("flower_data", exist_ok=True)
dataset = OxfordFlowersDataset("./flower_data")
print(f"Total samples: {len(dataset)}")

img, label = dataset[0]

