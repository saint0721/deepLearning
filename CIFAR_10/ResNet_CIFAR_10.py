import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import wandb
from customDataLoader import CIFAR_DataLoader, custom_transform
from ResNet import ResNet18
from utils import load_train_cifar_data, load_test_cifar_data

# wandb 초기화
wandb.init(
    project="ResNet-CIFAR10",
    config={
        "learning_rate": 0.1,
        "architecture": "ResNet",
        "dataset": "CIFAR-10",
        "epochs": 10,
    },
)
wandb.run.name = "ResNet18 inference"

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

# 폴더 지정 & 데이터로더 설정
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


# 학습률 조정
def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# 학습 함수
def train(epoch):
    print(f"\n[ Train epoch: {epoch} ]")
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
                f"\nBatch {batch_idx}: Accuracy {correct / total:.4f}, Loss {train_loss / total:.4f}"
            )

    avg_accuracy = correct / total
    avg_loss = train_loss / total
    grad_norm = sum(p.grad.data.norm(2).item() for p in net.parameters())
    print(f"\nTrain Epoch {epoch}: Accuracy {avg_accuracy:.4f}, Loss {avg_loss:.4f}")
    wandb.log(
        {
            "train accuracy": avg_accuracy,
            "train loss": avg_loss,
            "grad_norm": grad_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        }
    )
    return avg_loss


# 평가 함수
def test(epoch):
    print(f"\n[ Test epoch: {epoch} ]")
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

    avg_accuracy = correct / total
    avg_loss = test_loss / total
    print(f"\nTest Epoch {epoch}: Accuracy {avg_accuracy:.4f}, Loss {avg_loss:.4f}")
    wandb.log({"test accuracy": avg_accuracy, "test loss": avg_loss, "epoch": epoch})

    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")
    torch.save({"net": net.state_dict()}, f"./checkpoint/{file_name}")
    print("Model Saved")

    return avg_loss


# 학습 및 평가 루프
for epoch in range(1, 10):
    adjust_learning_rate(optimizer, epoch)
    train_loss = train(epoch)
    current_test_loss = test(epoch)

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
