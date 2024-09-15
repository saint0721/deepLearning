import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(self.df['path'][index])
        label = torch.tensor(int(self.df['cell_type_idx'][index]))

        # 데이터 전처리
        if self.transform:
            image = self.transform(image)

        return image, label
