import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b7

# GPU 설정 (가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-100 데이터 로드 및 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet은 입력 크기가 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)  # 배치 크기 증가

# EfficientNet-B7 모델 불러오기 (미리 학습된 가중치를 사용하지 않음)
model = efficientnet_b7(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)  # CIFAR-100이 100개의 클래스를 가짐
model = model.to(device)

# 손실 함수 및 최적화기 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 무한 루프
epoch = 0
while True:
    epoch += 1
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()  # 기울기 초기화
        outputs = model(images)  # 모델에 데이터 입력
        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()  # 역전파 수행
        optimizer.step()  # 가중치 업데이트
        
        running_loss += loss.item()
    
    # 에폭마다 출력
    print(f'Epoch [{epoch}], Loss: {running_loss/len(train_loader):.4f}')
    
    # 주기적으로 모델 저장 (10,000 에폭마다)
    if epoch % 10000 == 0:
        torch.save(model.state_dict(), f'efficientnetb7_epoch_{epoch}.pth')