import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os


def unpickle(file):
    import pickle

    with open(file, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")
    return data_dict


class ModelTrainer:
    def __init__(self, model, classes, device="mps"):
        self.model = model
        self.classes = classes
        self.device = device
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def imshow(self, images, labels, predictions):
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.flatten()

        for img, label, pred, ax in zip(images, labels, predictions, axes):
            img = img / 2 + 0.5  # 정규화된 이미지를 원래 범위로 변환
            npimg = np.clip(img.numpy().transpose((1, 2, 0)), 0, 1)
            ax.imshow(npimg)
            ax.set_title(
                f"True: {self.classes[label]} \n Pred: {self.classes[pred]}",
                fontsize=20,
            )
            ax.axis("off")
        plt.tight_layout()
        plt.close()


def load_train_cifar_data(cifar_dir, batch_range):
    images_list = []
    labels_list = []
    for batch_number in batch_range:
        batch_file = os.path.join(cifar_dir, f"data_batch_{batch_number}")
        batch_data = unpickle(batch_file)
        images_list.append(batch_data[b"data"])
        labels_list.append(batch_data[b"labels"])
    image = np.vstack(images_list).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    label = np.hstack(labels_list)

    return image, label


def load_test_cifar_data(cifar_dir):
    images_list = []
    labels_list = []
    batch_file = os.path.join(cifar_dir, "test_batch")
    batch_data = unpickle(batch_file)
    images_list.append(batch_data[b"data"])
    labels_list.append(batch_data[b"labels"])

    image = np.vstack(images_list).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    label = np.hstack(labels_list)

    return image, label
