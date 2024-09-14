import torch
import torch.nn as nn
import torch.nn.functional as F


# Squeeze-and-Excitation (SE) Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        reduced_channels = in_channels // reduction_ratio
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        squeeze = F.adaptive_avg_pool2d(x, 1)
        excitation = torch.sigmoid(self.fc2(F.silu(self.fc1(squeeze))))
        return x * excitation


# MBConv Block (Mobile Inverted Bottleneck Conv Block)
class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expand_ratio,
        kernel_size,
        stride,
        se_ratio=4,
        drop_connect_rate=0.2,
    ):
        super(MBConvBlock, self).__init__()
        self.expand_ratio = expand_ratio  # 확장 비율
        self.use_residual = (
            stride == 1 and in_channels == out_channels
        )  # input.channels == output.channels가 되야지 잔차를 사용할 수 있음
        self.drop_connect_rate = drop_connect_rate

        expanded_channels = in_channels * expand_ratio

        # Expansion layer
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(
                in_channels, expanded_channels, kernel_size=1, bias=False
            )
            self.expand_bn = nn.BatchNorm2d(expanded_channels)

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            expanded_channels,
            expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expanded_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)

        # Squeeze and Excitation (SE) block
        self.se = SEBlock(expanded_channels, reduction_ratio=se_ratio)

        # Pointwise convolution (projection) 확장된 채널을 출력 채널로 축소
        self.project_conv = nn.Conv2d(
            expanded_channels, out_channels, kernel_size=1, bias=False
        )
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        # Expansion
        if self.expand_ratio != 1:
            x = F.relu6(self.expand_bn(self.expand_conv(x)))

        # Depthwise convolution
        x = F.relu6(self.depthwise_bn(self.depthwise_conv(x)))

        # Squeeze and Excitation
        x = self.se(x)

        # Pointwise convolution
        x = self.project_bn(self.project_conv(x))

        # Residual connection and drop connect
        if self.use_residual:
            if self.drop_connect_rate:
                x = self.drop_connect(x)
            x += identity
        return x

    def drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(
            [batch_size, 1, 1, 1], dtype=x.dtype, device=x.device
        )
        binary_tensor = torch.floor(
            random_tensor
        )  # floor은 소수점 아래를 무시, 이진으로 만듦

        x = x / keep_prob * binary_tensor
        return x


# EfficientNet Model
class EfficientNet(nn.Module):
    def __init__(
        self,
        num_classes=10,  # CIFAR10: 10, CIFAR:100, ImageNet: 1000으로 인자 변경하면 될듯
        width=1.0,
        depth=1.0,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        # Drop out과 비슷한 규제 방법, 훈련 중에 가중치를 제거.
        # 1 - drop_connect_rate = 유지될 확률
    ):
        super(EfficientNet, self).__init__()

        base_channels = 32
        final_channels = 640  # 원본: 1280 -> 640 or 960으로 변환
        out_channels = 16

        # Stem layer (input layer)
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU6(
                inplace=True
            ),  # 출력이 6을 넘지 않도록 하는 ReLU, 즉 숫자에 제한을 둠
            # 기본 ReLU는 출력에 제한이 없어서 x값이 커지면 ReLU값이 커진다.
        )

        # EfficientNet Blocks (stacked MBConvBlocks)
        self.blocks = nn.ModuleList([])  # 모듈을 리스트로 관리해주는 클래스
        self.blocks.append(
            MBConvBlock(
                base_channels, out_channels, expand_ratio=1, kernel_size=3, stride=1
            )
        )
        self.blocks.append(
            MBConvBlock(
                out_channels, out_channels * 6, expand_ratio=6, kernel_size=3, stride=2
            )
        )

        # Add more blocks depending on the configuration you need (e.g., B0 to B7)

        # Final layers (classifier)
        self.head = nn.Sequential(
            nn.Conv2d(out_channels * 6, final_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.ReLU6(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(final_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# EfficientNet B0 configuration
def efficientnet_b0(
    num_classes=10,
):  # 원본 모델은 ImageNet을 위해서 1000으로 설정해 놓음
    return EfficientNet(
        num_classes=num_classes,
        width=1.0,
        depth=1.0,
        dropout_rate=0.2,
        drop_connect_rate=0.4,
    )


def efficientnet_b1(num_classes=10):
    return EfficientNet(
        num_classes=num_classes,
        width=1.0,
        depth=1.1,
        dropout_rate=0.2,
        drop_connect_rate=0.4,
    )


def efficientnet_b2(num_classes=10):
    return EfficientNet(
        num_classes=num_classes,
        width=1.1,
        depth=1.2,
        dropout_rate=0.3,
        drop_connect_rate=0.4,
    )


def efficientnet_b3(num_classes=10):
    return EfficientNet(
        num_classes=num_classes,
        width=1.2,
        depth=1.4,
        dropout_rate=0.3,
        drop_connect_rate=0.4,
    )


def efficientnet_b4(num_classes=10):
    return EfficientNet(
        num_classes=num_classes,
        width=1.4,
        depth=1.8,
        dropout_rate=0.4,
        drop_connect_rate=0.4,
    )


def efficientnet_b5(num_classes=10):
    return EfficientNet(
        num_classes=num_classes,
        width=1.6,
        depth=2.2,
        dropout_rate=0.4,
        drop_connect_rate=0.4,
    )


def efficientnet_b6(num_classes=10):
    return EfficientNet(
        num_classes=num_classes,
        width=1.8,
        depth=2.6,
        dropout_rate=0.5,
        drop_connect_rate=0.4,
    )


def efficientnet_b7(num_classes=10):
    return EfficientNet(
        num_classes=num_classes,
        width=2.0,
        depth=3.1,
        dropout_rate=0.5,
        drop_connect_rate=0.4,
    )
