# 파이썬 라이브러리
import os
from glob import glob

# 파이토치 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T

# 내가 만든 라이브러리
from torch.utils.data import DataLoader
from customDataLoader import CustomDataset
from get_data import get_data
from utils import compute_mean_stdev, AverageMeter
from model import initialize_model, set_parameter_requires_grad
import wandb


# wandb 초기화
wandb.init(
    project="HAM10000",
    config={
        "learning_rate": 1e-3,
        "architecture": "ResNet",
        "dataset": "HAM10000",
        "epochs": 10,
    },
)
wandb.run.name = "ResNet inference"

# Device & 모델 설정 
model_name = 'ResNet'
print(f"start: {model_name}")

classes = 7
feature_extract = False
model_ft, input_size = initialize_model(model_name, classes, feature_extract, use_pretrained=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model_ft.to(device)

# 모델 불러오기
file_name = "ResNet.pth"
if os.path.isfile(f"./checkpoint/{file_name}"):
    checkpoint = torch.load(f"./checkpoint/{file_name}")
    model.load_state_dict(checkpoint["net"]) 

# 최적화 & 손실함수 설정
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss().to(device)

# 폴더 지정 & 데이터로더 설정
data_dir ="/home/students/cs/202121165/deep_learning/skin-cancer-prediction/data"
all_image_path = glob(os.path.join(data_dir, "*", "*.jpg"))
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
print(f"이미지: {len(all_image_path)}장")

# 평균, 표준편차 계산
norm_means, norm_stdevs = compute_mean_stdev(all_image_path)
df_train, df_val = get_data(data_dir, imageid_path_dict)


train_transform = T.Compose([
    T.Resize((input_size, input_size)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
    T.ToTensor(),
    T.Normalize(norm_means, norm_stdevs)
])

val_transform = T.Compose([
    T.Resize((input_size, input_size)),
    T.ToTensor(),
    T.Normalize(norm_means, norm_stdevs)
])

train_dataset = CustomDataset(df_train, transform=train_transform)
val_dataset = CustomDataset(df_val, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


total_loss_train, total_acc_train = [], []
# 학습
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    
    # AverageMeter 인스턴스 생성
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    
    curr_iter = (epoch - 1) * len(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        N = images.size(0)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        prediction = outputs.max(1, keepdim=True)[1]
        # 업데이트: 정확도와 손실을 업데이트
        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)
        train_loss.update(loss.item(), N)
        curr_iter += 1

        if (i+1) % 100 == 0:
            print(f"[epoch {epoch}], [iter {i+1} / {len(train_loader)}], [train loss {train_loss.avg:.5f}], [train acc {train_acc.avg:.5f}]")
            total_loss_train.append(train_loss.avg)
            total_acc_train.append(train_acc.avg)
    
    wandb.log(
        {
            "train accuracy": train_acc.avg,
            "train loss": train_loss.avg,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
    )

    return train_loss.avg, train_acc.avg


# 평가 함수
def validate(val_loader, model, criterion, optimizer, epoch):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            N = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            prediction = outputs.max(1, keepdim=True)[1]
            val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N, N)
            val_loss.update(loss.item(), N)

    print(f"[epoch {epoch}], [val loss {val_loss.avg:.5f}], [val acc {val_acc.avg:.5f}]")

    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")
    torch.save({"net": model.state_dict()}, f"./checkpoint/{file_name}")
    print("Model Saved")

    wandb.log(
        {
            "validation accuracy": val_loss.avg,
            "validation loss": val_acc.avg,
        }
    )

    return val_loss.avg, val_acc.avg

epoch = 10
best_val_acc = 0
total_loss_val, total_acc_val = [], []

# 학습 및 평가 루프
for epoch in range(1, epoch+1):
    train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
    loss_val, acc_val = validate(val_loader, model, criterion, optimizer, epoch)
    total_loss_val.append(loss_val)
    total_acc_val.append(acc_val)

    if acc_val > best_val_acc:
        best_val_acc = acc_val
        print('*****************************************************')
        print(f'best record: [epoch {epoch}], [val loss {loss_val:.5f}], [val acc {acc_val:.5f}]')
        print('*****************************************************')
    

wandb.finish()
