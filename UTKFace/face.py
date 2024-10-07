# 파이썬 라이브러리
import os
from glob import glob
from fvcore.nn import FlopCountAnalysis, flop_count_table

# 파이토치 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from torchvision.models import *

# 내가 만든 라이브러리
from torch.utils.data import DataLoader
from customDataLoader import CustomDataset
from get_data import get_data
from utils import compute_mean_stdev, AverageMeter
from model import initialize_model
import wandb

# wandb 초기화
wandb.init(
    project="UTKFace",
    config={
        "learning_rate": 1e-3,
        "architecture": "ResNet50",
        "dataset": "UTKFace",
        "epochs": 10,
    },
)
wandb.run.name = "ResNet50"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Device & 모델 설정 
model_name = 'ResNet'
print(f"start: {model_name}")

classes = 1
feature_extract = False

# 나이 예측 모델 설정 (회귀)
task_type = "regression"
model_ft_age, input_size_age = initialize_model(model_name, task_type, feature_extract, use_pretrained=False)
model_age = model_ft_age.to(device)

# 성별 예측 모델 설정 (분류)
task_type = "classification"
model_ft_gender, input_size_gender = initialize_model(model_name, task_type, feature_extract, use_pretrained=False)
model_gender = model_ft_gender.to(device)


# 나이 예측 모델 불러오기
file_name_age = "ResNet_age.pth"
if os.path.isfile(f"./checkpoint/{file_name_age}"):
    checkpoint_age = torch.load(f"./checkpoint/{file_name_age}")
    model_age.load_state_dict(checkpoint_age["net"]) 

# 성별 예측 모델 불러오기
file_name_gender = "ResNet_gender.pth"
if os.path.isfile(f"./checkpoint/{file_name_gender}"):
    checkpoint_gender = torch.load(f"./checkpoint/{file_name_gender}")
    model_gender.load_state_dict(checkpoint_gender["net"]) 

# 나이 최적화 & 손실함수 설정
optimizer_age = optim.Adam(model_age.parameters(), lr=1e-3)
criterion_age = nn.MSELoss()

# 성별 최적화 & 손실함수 설정
optimizer_gender = optim.Adam(model_gender.parameters(), lr=1e-3)
criterion_gender = nn.CrossEntropyLoss()

# 폴더 지정 & 데이터로더 설정
data_dir = "/home/work/bachelor/deep_learning/UTKFace/datasets/part1"

# 이미지 파일 확장자를 제한하지 않고 모든 파일을 읽음
all_image_path = glob(os.path.join(data_dir, "*.*"))

# 이미지 파일 경로 확인
if len(all_image_path) == 0:
    print("이미지 파일이 없습니다. 경로와 파일 확장자를 확인하세요.")
else:
    print(f"이미지: {len(all_image_path)}장")

# 평균, 표준편차 계산
norm_means, norm_stdevs = compute_mean_stdev(all_image_path)

# 데이터 로딩
(x_train_age, x_test_age, y_train_age, y_test_age), (x_train_gender, x_test_gender, y_train_gender, y_test_gender) = get_data(data_dir)

# 나이 정규화
y_train_age = y_train_age / 100.0
y_test_age = y_test_age / 100.0

# transform 적용
train_transform = T.Compose([
    T.Resize((input_size_age, input_size_age)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
    T.ToTensor(),
    T.Normalize(norm_means, norm_stdevs)
])

val_transform = T.Compose([
    T.Resize((input_size_age, input_size_age)),
    T.ToTensor(),
    T.Normalize(norm_means, norm_stdevs)
])

# 이제 나이 데이터셋과 성별 데이터셋을 각각의 데이터 로더에 할당 가능
train_dataset_age = CustomDataset(x_train_age, y_train_age, transform=train_transform)
val_dataset_age = CustomDataset(x_test_age, y_test_age, transform=val_transform)

train_loader_age = DataLoader(train_dataset_age, batch_size=8, shuffle=True)
val_loader_age = DataLoader(val_dataset_age, batch_size=8, shuffle=False)

# 성별 데이터셋 설정
train_dataset_gender = CustomDataset(x_train_gender, y_train_gender, transform=train_transform)
val_dataset_gender = CustomDataset(x_test_gender, y_test_gender, transform=val_transform)

train_loader_gender = DataLoader(train_dataset_gender, batch_size=8, shuffle=True)
val_loader_gender = DataLoader(val_dataset_gender, batch_size=8, shuffle=False)

# 나이 예측용 (회귀) run_epoch 함수
def run_epoch_age(loader, model, criterion, optimizer=None, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
    
    epoch_loss = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        N = images.size(0)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        # print(f"Outputs Shape: {outputs.shape}, Labels Shape: {labels.shape}")

        # 나이 예측 (회귀 문제일 경우 MSELoss 사용)
        # 출력과 타겟이 (batch_size, 1) 형태로 일치하도록 변환
        if outputs.dim() > 1 and outputs.shape[-1] != 1:
            outputs = outputs.view(-1, 1)
        if labels.dim() == 1:
            labels = labels.view(-1, 1)

        # 크기 불일치 확인
        if outputs.shape != labels.shape:
            print(f"Mismatch in shapes: Outputs {outputs.shape}, Labels {labels.shape}")

        loss = criterion(outputs, labels.float())

        if is_train:
            loss.backward()
            optimizer.step()

        # 평균 손실 업데이트
        epoch_loss.update(loss.item(), N)

    return epoch_loss.avg  # MSE 손실 값을 반환


# 성별 예측용 (분류) run_epoch 함수
def run_epoch_gender(loader, model, criterion, optimizer=None, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
    
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()  # 성별 예측 정확도 계산용

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        N = images.size(0)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        # print(f"Outputs Shape: {outputs.shape}, Labels Shape: {labels.shape}")

        # 성별 예측 (분류 문제일 경우 CrossEntropy 사용)
        loss = criterion(outputs, labels)
        prediction = outputs.max(1, keepdim=True)[1]
        accuracy = prediction.eq(labels.view_as(prediction)).sum().item() / N
        epoch_acc.update(accuracy, N)

        if is_train:
            loss.backward()
            optimizer.step()

        # 평균 손실 업데이트
        epoch_loss.update(loss.item(), N)

    return epoch_loss.avg, epoch_acc.avg  # 분류 문제에서는 손실과 정확도 값을 둘 다 반환

# 학습 및 평가 루프
best_val_acc_gender = 0
best_val_loss_age = float("inf")  # 나이 예측은 손실(MSE) 값이 작을수록 좋으므로 초기값을 무한대로 설정
epochs = 10
total_loss_train_age = []
total_loss_train_gender = []

total_loss_val_age = []
total_loss_val_gender = []

total_acc_train_gender = []
total_acc_val_gender = []

for epoch in range(1, epochs + 1):
    # 나이 데이터 학습 단계
    train_loss_age = run_epoch_age(train_loader_age, model_age, criterion_age, optimizer_age, is_train=True)
    total_loss_train_age.append(train_loss_age)

    # 나이 데이터 평가 단계
    val_loss_age = run_epoch_age(val_loader_age, model_age, criterion_age, optimizer=None, is_train=False)
    total_loss_val_age.append(val_loss_age)

    # 성별 데이터 학습 단계
    train_loss_gender, train_acc_gender = run_epoch_gender(train_loader_gender, model_gender, criterion_gender, optimizer_gender, is_train=True)
    total_loss_train_gender.append(train_loss_gender)
    total_acc_train_gender.append(train_acc_gender)

    # 성별 데이터 평가 단계
    val_loss_gender, val_acc_gender = run_epoch_gender(val_loader_gender, model_gender, criterion_gender, optimizer=None, is_train=False)
    total_loss_val_gender.append(val_loss_gender)
    total_acc_val_gender.append(val_acc_gender)

    # 결과 출력 (나이와 성별 데이터를 각각 출력)
    print(f"[Epoch {epoch}/{epochs}] Age Train Loss (MSE): {train_loss_age:.5f}, Age Val Loss (MSE): {val_loss_age:.5f}")
    print(f"[Epoch {epoch}/{epochs}] Gender Train Loss: {train_loss_gender:.5f}, Gender Train Acc: {train_acc_gender:.5f}, "
          f"Gender Val Loss: {val_loss_gender:.5f}, Gender Val Acc: {val_acc_gender:.5f}")

    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")

    # 나이 데이터에서 모델 저장 (최고 검증 손실(MSE) 낮은 값 기준으로 모델 저장)
    if val_loss_age < best_val_loss_age:
        best_val_loss_age = val_loss_age
        print('*****************************************************')
        print(f'New best model for age (MSE) saved at epoch {epoch}: Val Loss: {val_loss_age:.5f}')
        print('*****************************************************')
        torch.save({"net": model_age.state_dict()}, f"./checkpoint/best_age_{file_name_age}")

    # 성별 데이터에서 모델 저장 (최고 검증 정확도 기준으로 모델 저장)
    if val_acc_gender > best_val_acc_gender:
        best_val_acc_gender = val_acc_gender
        print('*****************************************************')
        print(f'New best model for gender saved at epoch {epoch}: Val Loss: {val_loss_gender:.5f}, Val Acc: {val_acc_gender:.5f}')
        print('*****************************************************')
        torch.save({"net": model_gender.state_dict()}, f"./checkpoint/best_gender_{file_name_gender}")

    # 파라미터 개수 출력
    model_parameters_age = sum(p.numel() for p in model_age.parameters() if p.requires_grad)
    model_parameters_gender = sum(p.numel() for p in model_gender.parameters() if p.requires_grad)

    # FLOPs 출력 (성별과 나이 예측 각각 별도로 계산 가능)
    input_img = torch.ones((1, 3, input_size_age, input_size_age)).to(device)
    
    # 나이 예측 모델 FLOPs 계산
    flops_age = FlopCountAnalysis(model_age, input_img)
    flops_total_age = flops_age.total() # 모델 전체 FlOPs 출력
    flops_table_age = flop_count_table(flops_age) # 테이블 형태로 각 연산하는 모듈마다 출력하고, 전체도 출력
    print("Age Model FLOPs:")
    print(flops_table_age)

    # 성별 예측 모델 FLOPs 계산
    input_img_gender = torch.ones((1, 3, input_size_gender, input_size_gender)).to(device)
    flops_gender = FlopCountAnalysis(model_gender, input_img_gender)
    flops_total_gender = flops_gender.total() # 모델 전체 FlOPs 출력
    flops_table_gender = flop_count_table(flops_gender) # 테이블 형태로 각 연산하는 모듈마다 출력하고, 전체도 출력
    print("Gender Model FLOPs:")
    print(flops_table_gender)

    # WandB 로그 기록
    wandb.log({
        "epoch": epoch,
        "age_train_loss": train_loss_age,  # 나이 예측의 손실 (MSE 손실이기도 하지만 명시적으로 기록 가능)
        "age_val_loss": val_loss_age,      # 나이 예측의 손실
        "gender_train_loss": train_loss_gender,
        "gender_train_accuracy": train_acc_gender,
        "gender_val_loss": val_loss_gender,
        "gender_val_accuracy": val_acc_gender,
        "best_val_loss_age": best_val_loss_age,
        "best_val_acc_gender": best_val_acc_gender,
        "model_parameters_age": model_parameters_age,
        "model_parameters_gender": model_parameters_gender,
        "flops_total_age": flops_total_age,
        "flops_total_gender": flops_total_gender,
    })

wandb.finish()
