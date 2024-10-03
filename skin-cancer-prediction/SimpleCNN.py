import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # (16, 32, 32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # (32, 16, 16)
        self.pool = nn.MaxPool2d(2, 2)  # (32, 8, 8)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # 수정된 입력 차원 (2048)
        self.fc2 = nn.Linear(128, 3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x