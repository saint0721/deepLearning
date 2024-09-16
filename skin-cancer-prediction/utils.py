# 파이썬 라이브러리
import numpy as np
import cv2
from tqdm import tqdm

# 파이토치 라이브러리
import torch
import torch.nn as nn


def compute_mean_stdev(image_paths):
    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    # tqdm: 진행도 확인할 수 있는 라이브러리ㅣ
    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i]) # BGR로 읽음
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)
    
    # (N, 224, 224, 3)으로 변환 axis=0 행, axis=1로 하면 열로 만들어짐
    imgs = np.stack(imgs, axis=0) 
    print(imgs.shape) # 이미지 객체 배열 확인
    imgs = imgs.astype(np.float32) / 255. # 픽셀값 정규화

    for i in range(3): # BGR 채널별로 계산
        pixels = imgs[:, :, :, i].ravel() # 1차원 배열로 변환
        means.append(np.mean(pixels)) # 평균 계산
        stdevs.append(np.std(pixels)) # 표준편차 계산
    
    means.reverse() # RGB로 변환
    stdevs.reverse() # RGB로 변환

    print(f"NormMean: {means}")
    print(f"normStdev: {stdevs}")
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