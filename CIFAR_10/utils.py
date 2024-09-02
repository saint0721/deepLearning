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


def train(net, train_loader, optimizer, criterion, device, epoch):
    print("\n[ Train epoch: %d ]" % epoch)
    net.train()
    train_loss, correct, total = 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(
                f"\nBatch {batch_idx} Accuracy {correct / total:.4f}, Loss {train_loss / total:.4f}"
            )

    train_accuracy = correct / total
    train_loss = train_loss / total
    grad_norm = sum(p.grad.data.norm(2).item() for p in net.parameters())
    print(
        f"\nTrain Epoch {epoch} Accuracy {train_accuracy:.4f}, Loss {train_loss:.4f}, grad_norm {grad_norm}"
    )


def test(net, test_loader, criterion, device, epoch, file_name):
    print(f"\n[ Test epoch: %d]" % epoch)
    net.eval()
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            test_loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_accuracy = correct / total
    test_loss = test_loss / total
    print(f"\nTest Epoch {epoch}: Accuracy {test_accuracy:.4f}, Loss {test_loss:.4f}")

    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")
    torch.save({"net": net.state_dict()}, f"./checkpoint/{file_name}")
    print("Model Saved")

    return test_loss


def adjust_learning_rate(optimizer, learning_rate, epoch):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


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
