import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from customDataLoader import CIFAR_DataLoader, custom_transform
from utils import ModelTrainer, load_train_cifar_data, load_test_cifar_data
import torch.optim as optim
import wandb

wandb.init(
    project="CIFAR10_upgrade",
    config={
        "learning_rate": 0.1,
        "architecture": "ResNet",
        "dataset": "CIFAR-10",
        "epochs": 10,
    },
)
wandb.run.name = "CIFAR_10 upgrade"


class basicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(basicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        out = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += self.residual(out)
        x = self.relu2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu3 = nn.ReLU()
        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        out = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += self.residual(out)
        x = self.relu3(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.conv3 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.conv4 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.conv5 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_layer()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Model:
    def resnet18(self):
        return ResNet(basicBlock, [2, 2, 2, 2])

    def resnet34(self):
        return ResNet(basicBlock, [3, 4, 6, 3])

    def resnet50(self):
        return ResNet(Bottleneck, [3, 4, 6, 3])

    def resnet101(self):
        return ResNet(Bottleneck, [3, 4, 23, 3])

    def resnet152(self):
        return ResNet(Bottleneck, [3, 8, 36, 3])


if __name__ == "__main__":
    cifar_dir = "/Users/SaintKim/Python/Artificial_Intelligence/Deep_Learning/CIFAR_10/cifar-10-batches-py"

    train_images, train_labels = load_train_cifar_data(cifar_dir, range(1, 6))
    test_images, test_labels = load_test_cifar_data(cifar_dir)

    train_dataset = CIFAR_DataLoader(
        train_images, train_labels, transform=custom_transform
    )
    test_dataset = CIFAR_DataLoader(
        test_images, test_labels, transform=custom_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # 모델 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = Model().resnet34().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = ModelTrainer(model, classes, device)

    # 모델 훈련 및 평가
    for i in range(1, 11):
        train = trainer.train(train_loader, optimizer, log_interval=200)
        test_loss, test_accuracy = trainer.evaluate(test_loader)

    wandb.log(
        {"train loss": train, "test loss": test_loss, "test accuracy": test_accuracy}
    )
    # 정확도 출력
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # 모델 저장
    torch.save(model, "cifar_10_model.pth")
