import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels.iloc[index]  # 여기를 iloc으로 수정하여 Pandas 인덱싱 문제 해결

        # 데이터 전처리
        if self.transform:
            image = self.transform(Image.fromarray(image))

        return image, label