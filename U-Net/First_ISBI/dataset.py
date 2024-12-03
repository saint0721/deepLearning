import torch
import os
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        lst_data = os.listdir(self.data_dir)
        
        lst_label = sorted([f for f in lst_data if f.startswith('label')])
        lst_input = sorted([f for f in lst_data if f.startswith('input')])
        
        if len(lst_label) != len(lst_input):
            raise ValueError("Number of labels and inputs do not match!")
        
        self.lst_label = lst_label
        self.lst_input = lst_input
        
    def __len__(self):
        return len(self.lst_label)
    
    def __getitem__(self, index):
        if index >= len(self.lst_label):
            raise IndexError(f"Index {index} is out of range for dataset of size {len(self.lst_label)}")
        
        label = np.load(os.path.join(self.data_dir, self.lst_label[index])).astype(np.float32) / 255.0
        input = np.load(os.path.join(self.data_dir, self.lst_input[index])).astype(np.float32) / 255.0
        
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]
        
        data = {'input': input, 'label': label}
        
        if self.transform:
            data = self.transform(data)
        
        return data