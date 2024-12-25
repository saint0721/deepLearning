# 파이썬 라이브러리
import os
import numpy as np
import matplotlib.pyplot as plt

# 파이토치 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim
from optimizer import get_optimizer
from torchvision import transforms

# 내가 만든 라이브러리
from torch.utils.data import DataLoader
from dataset import Dataset
from utils import compute_mean_stdev, AverageMeter, ToTensor, RandomFlip, Normalization
from model import initialize_model
import wandb

# wandb 초기화
wandb.init(
    project="U-Net",
    config={
        "learning_rate": 1e-3,
        "architecture": "U-Net",
        "dataset": "ISBI-challenge",
        "epochs": 10,
    },
)
wandb.run.name = "U-Net initiative"

# Device & 모델 설정 
model_name = 'UNet'
print(f"start: {model_name}")

classes = 1
feature_extract = False
model_ft, input_size = initialize_model(model_name, classes, feature_extract, use_pretrained=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate=1e-3
model = model_ft.to(device)

# 모델 불러오기
file_name = "efficientnet.pth"
if os.path.isfile(f"./checkpoint/{file_name}"):
    checkpoint = torch.load(f"./checkpoint/{file_name}")
    model.load_state_dict(checkpoint["net"]) 

# 최적화 & 손실함수 설정
optimizer_name = 'Adam'
optimizer = get_optimizer(optimizer_name, model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss().to(device)

# 폴더 지정 & 데이터로더 설정
dir_data = "./dataset"
base_dir = '/home/students/cs/202121165/public_html/deepLearning/U-Net/dataset/train'
data_dir = dir_data

train_input_path = glob(os.path.join(data_dir, "train", "input_*.npy"))
train_label_path = glob(os.path.join(data_dir, "train", "label_*.npy"))
test_input_path = glob(os.path.join(data_dir, "val", "input_*.npy"))
test_label_path = glob(os.path.join(data_dir, "val", "label_*.npy"))
val_input_path = glob(os.path.join(data_dir, "val", "input_*.npy"))
val_label_path = glob(os.path.join(data_dir, "val", "label_*.npy"))

train_imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in train_input_path}
test_imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in test_input_path}
val_imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in val_input_path}
print(f" train input: {len(train_input_path)}장")
print(f" test input: {len(test_input_path)}장")
print(f" val input: {len(val_input_path)}장")

# 평균, 표준편차 계산
norm_means, norm_stdevs = compute_mean_stdev(train_input_path)
norm_means, norm_stdevs = compute_mean_stdev(test_input_path)
norm_means, norm_stdevs = compute_mean_stdev(val_input_path)

transform = transforms.Compose([ RandomFlip(), ToTensor(), Normalization(mean=norm_means, std=norm_stdevs) ])
dir_save_train = os.path.join(dir_data, 'train')

dataset_train = Dataset(dir_save_train, transform=transform)
data = dataset_train.__getitem__(0)
input = data['input']
label = data['label']
print(f"Images Shape: {input.shape}, Labels Shape: {label.shape}")
print(f"Images Type: {type(input)}, Labels Type: {type(label)}")

# 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
      "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

# 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
    
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_lst.sort(lambda f: int("".join(filter(str.isdigit, f))))
    
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    
    net.load_state_dict(dict_model(['net']))
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('pth')[0])
    
    return net, optim, epoch

# 훈련 파라미터 설정하기
batch_size = 4
num_epoch = 20

base_dir = '/home/students/cs/202121165/public_html/deepLearning/U-Net/dataset'
data_dir = dir_data
ckpt_dir = os.path.join(base_dir, "checkpoint")

transform = transforms.Compose([ RandomFlip(), ToTensor(), Normalization(mean=norm_means, std=norm_stdevs) ])

dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform, )
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'),
transform=transform)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

# Optimizer& 손실함수 정의하기
criterion = nn.BCEWithLogitsLoss().to(device)
optim = optim.Adam(model.parameters(), lr=1e-3)

# 그밖에 부수적인 variables 설정하기
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

# 그 밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

# 네트워크 학습시키기
st_epoch = 0 # Start epoch
ckpt_dir = os.path.join(base_dir, "checkpoint") # ckpt: CheckPoint

# 학습 루프
def train(train_loader, val_loader, model, criterion, optimizer, start_epoch, num_epoch, ckpt_dir):
    avg_train_loss = None  # 평균 학습 손실 초기화
    for epoch in range(start_epoch + 1, num_epoch + 1):
        # 학습 모드
        model.train()
        train_loss_arr = []

        for batch, data in enumerate(train_loader, 1):
            # 데이터 로드
            label = data['label'].to(device)
            input = data['input'].to(device)

            # Forward pass
            output = model(input)

            # Backward pass
            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # 손실 저장
            train_loss_arr.append(loss.item())

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, len(train_loader), torch.mean(torch.tensor(train_loss_arr))))

        avg_train_loss = torch.mean(torch.tensor(train_loss_arr))  # 평균 학습 손실 계산

        # 검증 루프
        val_loss_arr = []
        model.eval()
        with torch.no_grad():
            for batch, data in enumerate(val_loader, 1):
                # 데이터 로드
                input = data['input'].to(device)
                label = data['label'].to(device)

                # Forward pass
                output = model(input)

                # 손실 계산
                loss = criterion(output, label)
                val_loss_arr.append(loss.item())

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, len(val_loader), torch.mean(torch.tensor(val_loss_arr))))

        # 모델 저장: 매 50 에폭마다 또는 검증 손실이 개선된 경우
        avg_val_loss = torch.mean(torch.tensor(val_loss_arr))
        if epoch % 50 == 0 or (epoch == start_epoch + 1 or avg_val_loss < min(val_loss_arr)):
            save(ckpt_dir=ckpt_dir, net=model, optim=optimizer, epoch=epoch)
            print(f"Model saved at epoch {epoch}")

        # wandb 기록
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch": epoch
        })

    return avg_train_loss  # 평균 학습 손실 반환

# 평가 함수
def validate(val_loader, model, criterion, start_epoch, num_epoch):
    model.eval()
    val_loss_arr = []

    with torch.no_grad():
        for batch, data in enumerate(val_loader, 1):
            # 데이터 로드
            inputs = data['input'].to(device)
            labels = data['label'].to(device)

            # Forward pass
            outputs = model(inputs)

            # 손실 계산
            loss = criterion(outputs, labels)
            val_loss_arr.append(loss.item())

            print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, len(val_loader), torch.mean(torch.tensor(val_loss_arr))))

    avg_val_loss = torch.mean(torch.tensor(val_loss_arr))

    # wandb 기록
    wandb.log({
        "val_loss": avg_val_loss,
        "epoch": epoch
    })

    return avg_val_loss


# 학습 및 평가 루프
epoch = 10
best_val_loss = float('inf')

for epoch in range(1, epoch + 1):
    train_loss = train(train_loader, val_loader, model, criterion, optimizer, st_epoch, num_epoch, ckpt_dir)
    val_loss = validate(val_loader, model, criterion, st_epoch, num_epoch)

    # 모델 저장: 매 50 에폭마다 또는 검증 손실이 개선된 경우
    if epoch % 50 == 0 or val_loss < best_val_loss:
        best_val_loss = val_loss
        save(ckpt_dir=ckpt_dir, net=model, optim=optimizer, epoch=epoch)
        print(f"Model saved at epoch {epoch}")

    print(f"Epoch {epoch}/{num_epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

wandb.finish()
