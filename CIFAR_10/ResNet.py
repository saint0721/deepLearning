import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        # in_planes: input planes
        super(BasicBlock, self).__init__()

        # 1 번째 합성곱 레이어
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)  # 배치 정규화
        self.relu = nn.ReLU(inplace=True)  # 활성화 함수
        self.conv2 = nn.Conv2d(  # 2 번째 합성곱 레이어
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut 경로 초기화
        self.shortcut = nn.Sequential()
        # Shortcut(Skip connection)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),  # 1X1 합성곱으로 차원 맞춤
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(
            self.bn1(self.conv1(x))
        )  # 첫 번째 합성곱 -> 배치 정규화 -> ReLU
        out = self.bn2(self.conv2(out))  # 두 번째 합성곱 -> 배치 정규화
        # skip connection + x = F(x) + x
        out += self.shortcut(x)
        out = self.relu(out)  # 최종 ReLU 적용
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 64개의 3x3 필터(filter)를 사용
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels  # 다음 레이어를 위해 채널 수 변경
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # 출력: [batch_size, 512, 4, 4]
        out = F.avg_pool2d(out, 4)  # 출력: [batch_size, 512, 1, 1]
        out = out.view(out.size(0), -1)  # 출력: [batch_size, 512]
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
