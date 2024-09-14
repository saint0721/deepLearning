import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as T
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
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                ]
            )
        elif phase == "test":
            return T.Compose(
                [
                    T.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # OpenCV로 전처리
        image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_pil = Image.fromarray(image_cv)

        # 데이터 전처리
        if self.transform:
            image_pil = self.transform(image_pil)

        return image_pil, torch.tensor(label, dtype=torch.long)


def custom_transform(image):
    image_np = np.array(image)
    image_resize = cv2.resize(image_np, (32, 32))
    image_nor = image_resize / 255.0
    image_transpose = np.transpose(image_nor, (2, 0, 1))

    return torch.tensor(image_transpose, dtype=torch.float)
