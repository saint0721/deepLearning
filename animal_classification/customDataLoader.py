import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

class AnimalImageDataset(Dataset):
    def __init__(self, root_dir, is_test=False, transform=None):
        self.root_dir = root_dir
        self.is_test = is_test
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.root_dir, image_file)
        image = Image.open(image_path).convert("RGB")

        if self.is_test:
            label = -1
        else:
            image_label = image_file.split('.')[0]
            label = 0 if 'cat' in image_label.lower() else 1
            label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label