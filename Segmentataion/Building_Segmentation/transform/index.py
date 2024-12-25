# 파이썬 라이브러리
import os
import warnings
import time
warnings.filterwarnings('ignore')

# 파이토치 라이브러리
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# 내가 만든 라이브러리
from dataset import BuildingsDataset
from get_data import get_class_label
from augmentation import get_training_augmentation, get_preprocessing, get_validation_augmentation
from utils import F1Score
from visualize import plot_metrics
import wandb

# wandb 초기화
wandb.init(
    project="U-Net",
    config={
        "learning_rate": 1e-3,
        "architecture": "U-Net",
        "dataset": "Massachussetts Building",
        "epochs": 50,
    },
)
wandb.run.name = "U-Net Building Segmentation"

# 폴더지정 & 데이터로더 설정
dir_data = "/home/saint/deepLearning/U-Net/Building_Segmentation/massachusetts-buildings-dataset/tiff"
x_train_dir = os.path.join(dir_data, 'train')
y_train_dir = os.path.join(dir_data, 'train_labels')
x_valid_dir = os.path.join(dir_data, 'val')
y_valid_dir = os.path.join(dir_data, 'val_labels')
x_test_dir = os.path.join(dir_data, 'test')
y_test_dir = os.path.join(dir_data, 'test_labels')

# 클래스 라벨 불러오기
select_class_rgb_values, select_classes = get_class_label("/home/saint/deepLearning/U-Net/Building_Segmentation/massachusetts-buildings-dataset/label_class_dict.csv")

# 데이터 증강 & 시각화
augmented_dataset = BuildingsDataset(x_train_dir, y_train_dir, augmentation=get_training_augmentation(), class_rgb_values=select_class_rgb_values)

# 모델 설정
device = "cuda" if torch.cuda.is_available() else "cpu"  
Training = True
epochs = 50

encoder = 'resnet50' # Backbone으로 ResNet50 불러오기
encoder_weights = 'imagenet' # ImageNet사전 훈련 
classes = select_classes # class: Background, Building
activation = 'sigmoid' # 활성화 함수로 Sigmoid 선택

model = smp.Unet(
    encoder_name=encoder,
    encoder_weights=encoder_weights,
    classes=len(classes),
    activation=activation
)
print(f"start: {model}")

# 전처리 함수
preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)   

# Metrics 정의
loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Precision(threshold=0.5),
    smp.utils.metrics.Recall(threshold=0.5),
    smp.utils.metrics.Accuracy(threshold=0.5),
    F1Score(beta=1, eps=1e-7)
]

# Optimizer & Scheduler 설정
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5)

# 가중치 로드
if os.path.exists('/home/saint/deepLearning/U-Net/Building_Segmentation/transform/best_model/best_model.pth'):
    model.load_state_dict(torch.load('/home/saint/deepLearning/U-Net/Building_Segmentation/transform/best_model/best_model.pth', map_location=device))
    print("Model loaded successfully")
    
# 데이터로더
train_dataset = BuildingsDataset(
    x_train_dir, y_train_dir, augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values
)
valid_dataset = BuildingsDataset(
	x_valid_dir, y_valid_dir, augmentation=get_validation_augmentation(),
	preprocessing=get_preprocessing(preprocessing_fn),
	class_rgb_values = select_class_rgb_values
)
test_dataset = BuildingsDataset(
	x_valid_dir, y_valid_dir, augmentation=get_validation_augmentation(),
	preprocessing=get_preprocessing(preprocessing_fn),
	class_rgb_values = select_class_rgb_values
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 훈련
Train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=device,
    verbose=True, # 학습중 상세정보 로깅
)

# 검증
Valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=device,
    verbose=True,
)

# CPU Time 측정
start_time = time.time()

# 학습 진행
if Training:
  best_IoU_score = 0.0
  train_logs_list, valid_logs_list = [], []

  for i in range(0, epochs):
    print(f'\nEpoch: {i+1}')
    train_logs = Train_epoch.run(train_loader)
    valid_logs = Valid_epoch.run(valid_loader)
    train_logs_list.append(train_logs)
    valid_logs_list.append(valid_logs)

    if best_IoU_score < valid_logs['iou_score']:
        best_IoU_score = valid_logs['iou_score']
        torch.save(model.state_dict(), './best_model.pth')
        print('Model saved!')

end_time = time.time()
print(f"CPU Time: {end_time-start_time:.2f} seconds")

# 예측 결과 저장할 폴더 생성
sample_preds_folders = 'sample_predictions/'
if not os.path.exists(sample_preds_folders):
	os.makedirs(sample_preds_folders)
 
models = {
    "U-Net": smp.Unet(encoder_name='resnet50', encoder_weights='imagenet', classes=len(classes), activation=activation),
    "U-Net++": smp.UnetPlusPlus(encoder_name='resnet50', encoder_weights='imagenet', classes=len(classes), activation=activation),
    "DeepLabV3": smp.DeepLabV3(encoder_name='resnet50', encoder_weights='imagenet', classes=len(classes), activation=activation),
    "FPN": smp.FPN(encoder_name='resnet50', encoder_weights='imagenet', classes=len(classes), activation=activation),
}

train_epochs = {}
valid_epochs = {}

# 각 모델의 optimizer 설정
optimizers = {
    name: torch.optim.Adam(model.parameters(), lr=0.0001) for name, model in models.items()
}

# 각 모델의 lr & scheduler 설정
lr_schedulers = {
    name: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5) for name, optimizer in optimizers.items()
}

# 각 모델의 metrics 설정
metrics = {
    name: [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Precision(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.Accuracy(threshold=0.5),
        F1Score(beta=1, eps=1e-7)
    ] for name in models.keys()
}

# 각 모델의 loss 설정
losses = {
    name: smp.utils.losses.DiceLoss() for name in models.keys()
}

# 각 모델의 평가를 저장할 리스트 생성
train_logs_list = {name: [] for name in models.keys()}
valid_logs_list = {name: [] for name in models.keys()}

# train / valid 학습
for name, model in models.items():
    model = model.to(device)
    train_epochs[name] = smp.utils.train.TrainEpoch(
        model,
        loss=losses[name],
        metrics=metrics[name],
        optimizer=optimizers[name],
        device=device,
        verbose=True,
    )
    valid_epochs[name] = smp.utils.train.ValidEpoch(
        model,
        loss=losses[name],
        metrics=metrics[name],
        device=device,
        verbose=True,
    )

    if Training:
        best_iou_score = 0.0
        for i in range(0, epochs):
            print(f'\n{name} - Epoch: {i+1}')
            train_logs = train_epochs[name].run(train_loader)
            valid_logs = valid_epochs[name].run(valid_loader)
            train_logs_list[name].append(train_logs)
            valid_logs_list[name].append(valid_logs)

            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                torch.save(model, f'{name}_best_model.pth')
                print('Model saved!')

# test 평가
for name, model in models.items():
    best_model = torch.load(f'{name}_best_model.pth', map_location=device)
    test_epoch = smp.utils.train.ValidEpoch(
        best_model,
        loss=losses[name],
        metrics=metrics[name],
        device=device,
        verbose=True,
    )
    valid_logs = test_epoch.run(test_loader)
    print(f"\nEvaluation on Test Data for {name}: ")
    print(f"Mean F1 Score: {valid_logs['F1Score']:.4f}")
    print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
    print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")
    print(f"Mean Precision: {valid_logs['precision']:.4f}")
    print(f"Mean Accuracy: {valid_logs['accuracy']:.4f}")
    print(f"Mean Recall: {valid_logs['recall']:.4f}")    
    
# plot 그리기
for name, model in models.items():
    best_model = torch.load(f'{name}_best_model.pth', map_location=device)
    test_epoch = smp.utils.train.ValidEpoch(
        best_model,
        loss=losses[name],
        metrics=metrics[name],
        device=device,
        verbose=True,
    )
    valid_logs = test_epoch.run(test_loader)
    print(f"\nEvaluation on Test Data for {name}: ")
    for metric, value in valid_logs.items():
        print(f"Mean {metric}: {value:.4f}")
    
    plot_metrics(name, train_logs_list[name], valid_logs_list[name], valid_logs.keys())

# Wandb 로그
wandb.log({
    "Mean F1 Score": round(valid_logs['F1Score'], 4),
    "Mean IoU Score": round(valid_logs['iou_score'], 4),
    "Mean Dice Loss": round(valid_logs['dice_loss'], 4),
    "Mean Precision": round(valid_logs['precision'], 4),
    "Mean Accuracy": round(valid_logs['accuracy'], 4),
    "Mean Recall": round(valid_logs['recall'], 4)
})