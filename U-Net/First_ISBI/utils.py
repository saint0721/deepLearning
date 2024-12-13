# 파이썬 라이브러리
import pandas as pd
import numpy as np
import os, cv2
from tqdm import tqdm
import torch

def compute_mean_stdev(npy_paths):
    imgs = []
    means, stdevs = [], []

    for path in npy_paths:
        img = np.load(path).astype(np.float32) / 255.0
        imgs.append(img)

    imgs = np.stack(imgs, axis=0)  # (N, H, W, C) 형태로 스택
    for i in range(imgs.shape[-1]):  # 채널별로 처리
        pixels = imgs[..., i].ravel()  # 1D 배열로 변환
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    
    # print(f"NormMean: {means}")
    # print(f"NormStdev: {stdevs}")
    return means, stdevs
        
class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Transform 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)
        
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}
        
        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        
    def __call__(self, data):
        label, input = data['label'], data['input']
        input = (input - self.mean) / self.std
        data = {'label': label, 'input': input}
        
        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)
            
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)
        
        data = {'label': label, 'input': input}
        
        return data
