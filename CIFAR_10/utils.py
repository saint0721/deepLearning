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
            img = img / 2 + (0.5).clip(0, 1)  # 정규화된 이미지를 원래 범위로 변환
            npimg = img.numpy().transpose((1, 2, 0))
            ax.imshow(npimg)
            ax.set_title(
                f"True: {self.classes[label]} \n Pred: {self.classes[pred]}",
                fontsize=20,
            )
            ax.axis("off")
        plt.tight_layout()
        plt.close()

    def train(self, train_data, optimizer, log_interval, epoch):
        self.model.train()
        for batch_idx, (images, labels) in enumerate(train_data):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(f"[Epoch {epoch}] [Train Loss: {loss.item():.4f}]")

    def evaluate(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        images_to_show = []
        labels_to_show = []
        predictions_to_show = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                batch_loss = self.criterion(output, labels)  # 현재 배치의 손실
                test_loss += batch_loss.item() * labels.size(0)  # 배치 크기로 가중 평균
                _, prediction = output.max(1)  # 예측 결과 인덱스
                correct += prediction.eq(labels).sum().item()
                total += labels.size(0)

                if len(images_to_show) < 10:
                    images_to_show.extend(images.cpu().numpy())
                    labels_to_show.extend(labels.cpu().numpy())
                    predictions_to_show.extend(prediction.cpu().numpy().flatten())

                if len(images_to_show) >= 10:
                    break

        test_loss /= total
        test_accuracy = 100.0 * correct / total

        images_to_show = np.array(images_to_show)[:10]
        labels_to_show = np.array(labels_to_show)[:10]
        predictions_to_show = np.array(predictions_to_show)[:10]

        self.imshow(torch.tensor(images_to_show), labels_to_show, predictions_to_show)

        return test_loss, test_accuracy


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
