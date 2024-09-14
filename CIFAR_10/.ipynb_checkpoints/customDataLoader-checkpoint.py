import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as T
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from PIL import Image
import numpy as np


class CIFAR_DataLoader(Dataset):
    def __init__(self, images, labels, transform=None, phase="train"):
        self.images = images
        self.labels = labels
        self.transform = (
            transform if transform is not None else self.get_default_transform(phase)
        )
        self.phase = phase

    def get_default_transform(self, phase):
        if phase == "train":
            return T.Compose(
                [
                    T.Resize((32, 32)),
                    AutoAugment(
                        policy=AutoAugmentPolicy.IMAGENET
                    ),  # 학습용 데이터 증강
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        elif phase == "test":
            return T.Compose(
                [
                    T.Resize((32, 32)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # OpenCV로 전처리
        # image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_pil = Image.fromarray(image)

        # 데이터 전처리
        if self.transform:
            image_pil = self.transform(image_pil)

        return image_pil, torch.tensor(label, dtype=torch.long)
