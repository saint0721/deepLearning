import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import wandb
from customDataLoader import CIFAR_DataLoader, custom_transform
from ResNet import ResNet18
from utils import (
    train,
    test,
    load_train_cifar_data,
    load_test_cifar_data,
    adjust_learning_rate,
)

# wandb 초기화
wandb.init(
    project="CIFAR10 Classification",
    config={
        "learning_rate": 0.1,
        "architecture": "ResNet",
        "dataset": "CIFAR-10",
        "epochs": 10,
    },
)
wandb.run.name = "ResNet18 CIFAR_10 classification"

# Device 설정
device = "mps" if torch.backends.mps.is_available() else "cpu"
net = ResNet18().to(device)

# 모델 불러오기
file_name = "resnet18_cifar10.pth"
if os.path.isfile(f"./checkpoint/{file_name}"):
    checkpoint = torch.load(f"./checkpoint/{file_name}")
    net.load_state_dict(checkpoint["net"])

# 최적화 & 손실함수 설정
learning_rate = 0.1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002
)

# 폴더 지정
cifar_dir = "/Users/SaintKim/Python/Artificial_Intelligence/Deep_Learning/CIFAR_10/cifar-10-batches-py"
train_images, train_labels = load_train_cifar_data(cifar_dir, range(1, 6))
test_images, test_labels = load_test_cifar_data(cifar_dir)

train_dataset = CIFAR_DataLoader(
    train_images, train_labels, phase="train", transform=custom_transform
)
test_dataset = CIFAR_DataLoader(
    test_images, test_labels, phase="test", transform=custom_transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 조기 종료(early stopping)
early_stopping_epochs = 5
best_loss = float("inf")
early_stop_counter = 0

# 시간 측정
start_time = time.time()


# 학습 및 평가 루프
for epoch in range(1, 10):
    adjust_learning_rate(
        optimizer, learning_rate, epoch
    )  # 학습률 조정 시 learning_rate 전달
    train(net, train_loader, optimizer, criterion, device, epoch)
    current_test_loss = test(net, test_loader, criterion, device, epoch, file_name)

    # 조기 종료 로직
    if current_test_loss > best_loss:
        early_stop_counter += 1
    else:
        best_loss = current_test_loss
        early_stop_counter = 0

    if early_stop_counter >= early_stopping_epochs:
        print("Early stopping")
        break

    wandb.log({"early stopping": early_stop_counter})

    print(f"\nTime elapsed: {time.time() - start_time:.2f} seconds")

wandb.finish()
