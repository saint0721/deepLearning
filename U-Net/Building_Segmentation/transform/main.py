# 파이썬 라이브러리
import os, cv2
import random, tqdm
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
warnings.filterwarnings('ignore')

# 파이토치 라이브러리
import torch

from optimizer import get_optimizer
from torch.utils.data import DataLoader
import segmentation_models_pytorch.utils
import segmentation_models_pytorch as smp

# 내가 만든 라이브러리
from dataset import BuildingsDataset
from model import initialize_model
from get_data import get_class_label
from visualize import visualize, one_hot_encode, reverse_one_hot, colour_code_segmentation
from augmentation import get_training_augmentation, get_preprocessing, get_validation_augmentation
from utils import crop_image
import wandb

# wandb 초기화
wandb.init(
    project="U-Net",
    config={
        "learning_rate": 1e-3,
        "architecture": "U-Net",
        "dataset": "Massachussetts Building",
        "epochs": 12,
    },
)
wandb.run.name = "U-Net Building Segmentation"

# Device & 모델 설정 
model_ft, input_size, model_name = initialize_model("UNet", num_classes=3, feature_extract=False, use_pretrained=False)
print(f"start: {model_name}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model_ft.to(device)
Training = True

# 가중치 로드
if os.path.exists('/home/saint/deepLearning/U-Net/Building_Segmentation/best_model.pth'):
    model.load_state_dict(torch.load('/home/saint/deepLearning/U-Net/Building_Segmentation/best_model.pth', map_location=device))
    print("Model loaded successfully")
print("Model loaded successfully")

# Metrics 정의
criterion = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5)]

# Optimizer 설정
optimizer_name = 'Adam'
optimizer = get_optimizer(optimizer_name, model.parameters(), lr=0.00008)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5)

# 폴더지정 & 데이터로더 설정
dir_data = "/home/saint/deepLearning/U-Net/Building_Segmentation/massachusetts-buildings-dataset/tiff"
# 학습 데이터
x_train_dir = os.path.join(dir_data, 'train')
y_train_dir = os.path.join(dir_data, 'train_labels')

# 검증 데이터
x_valid_dir = os.path.join(dir_data, 'val')
y_valid_dir = os.path.join(dir_data, 'val_labels')

# 테스트 데이터
x_test_dir = os.path.join(dir_data, 'test')
y_test_dir = os.path.join(dir_data, 'test_labels')
select_class_rgb_values, select_classes = get_class_label("/home/saint/deepLearning/U-Net/Building_Segmentation/massachusetts-buildings-dataset/label_class_dict.csv")

# 데이터셋에서 샘플 가져오기
dataset = BuildingsDataset(x_train_dir, y_train_dir, class_rgb_values=select_class_rgb_values)

# 랜덤 샘플 선택
random_idx = random.randint(0, len(dataset) - 1)
image, mask = dataset[random_idx]  

# 랜덤 인덱스 시각화
visualize(
    original_image=image,
    ground_truth_image=colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
    one_hot_encoded_mask=reverse_one_hot(mask)
)

# 데이터 증강 & 시각화
augmented_dataset = BuildingsDataset(
    x_train_dir, y_train_dir, augmentation=get_training_augmentation(), class_rgb_values=select_class_rgb_values
)
random_idx = random.randint(0, len(augmented_dataset)-1)

for i in range(3):
    image, mask = augmented_dataset[random_idx]
    visualize(
    original_image=image,
    ground_truth_image=colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
    one_hot_encoded_mask=reverse_one_hot(mask)
)

# 데이터로더
train_dataset = BuildingsDataset(
    x_train_dir, y_train_dir, augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn=None),
    class_rgb_values=select_class_rgb_values
)
valid_dataset = BuildingsDataset(
	x_valid_dir, y_valid_dir, augmentation=get_validation_augmentation(),
	preprocessing=get_preprocessing(preprocessing_fn=None),
	class_rgb_values = select_class_rgb_values
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)

# 학습 설정
Train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=criterion, 
    metrics=metrics, 
    optimizer=optimizer,
    device=device,
    verbose=True, # 학습중 상세정보 로깅
)

# 검증
Valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=criterion, 
    metrics=metrics, 
    device=device,
    verbose=True,
)

# CPU Time 측정
start_time = time.time()

# 훈련 시작
if Training:
  best_IoU_score = 0.0
  train_logs_list, valid_logs_list = [], []

  for i in range(0, 12):
    print(f'\nEpoch: {i+1}')
    train_logs = Train_epoch.run(train_loader)
    valid_logs = Valid_epoch.run(valid_loader)
    train_logs_list.append(train_logs)
    valid_logs_list.append(valid_logs)

    if best_IoU_score < valid_logs['iou_score']:
        best_IoU_score = valid_logs['iou_score']
        torch.save(model.state_dict(), '/home/saint/deepLearning/U-Net/Building_Segmentation/best_model.pth')
        print('Model saved!')

end_time = time.time()
print(f"CPU Time: {end_time-start_time:.2f} seconds")

# 사전에 학습된 모델이 존재하면 load        
if os.path.exists('../best_model.pth'):
	best_model = torch.load('../best_model.pth', map_location=device)
	print("loaded UNet model from this run")
elif os.path.exists('../Building_Segmentation/best_model.pth'):
	best_model = torch.load('../Building_Segmentation/best_model.pth', map_location=device)
	print("loaded UNet model from previous commit")
 
# 테스트 
test_dataset = BuildingsDataset(
	x_test_dir, y_test_dir, augmentation=get_validation_augmentation(),
	preprocessing=get_preprocessing(preprocessing_fn=None), 
	class_rgb_values = select_class_rgb_values
)
test_dataloader = DataLoader(test_dataset)

test_dataset_vis = BuildingsDataset(
	x_test_dir, y_test_dir, augmentation=get_validation_augmentation(),
	class_rgb_values = select_class_rgb_values
)

random_idx = random.randint(0, len(test_dataset_vis)-1)
image, mask = test_dataset_vis[random_idx]

visualize(
	original_image = image, 
	ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
	one_hot_encoded_mask = reverse_one_hot(mask)
)

# 예측 결과 저장할 폴더 생성
sample_preds_folders = 'sample_predictions/'
if not os.path.exists(sample_preds_folders):
	os.makedirs(sample_preds_folders)
 
for idx in range(len(test_dataset)):
	random_idx = random.randint(0, len(test_dataset)-1)
	image, gt_mask = test_dataset[random_idx]
	image_vis = crop_image(test_dataset_vis[random_idx][0].astype('uint8'))
	x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)

	# predict test iamge
	pred_mask = best_model(x_tensor)
	pred_mask = pred_mask.detach().squeeze().cpu().numpy()

	# convert pred_mask CHW -> HWC
	pred_mask = np.transpose(pred_mask, (1, 2, 0))

	# Get prediction 
	pred_building_heatmap = pred_mask[:, :, select_classes.index('building')]
	pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values))

	# convert gt_mask
	gt_mask = np.transpose(gt_mask, (1, 2, 0))
	gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values))
	cv2.imwrite(os.path.join(sample_preds_folders, f"sample_pred_{idx}.png"), np.hstack([image_vis, gt_mask, pred_mask])[:, :, ::-1])

	visualize(
		original_image = image_vis,
		ground_truth_mask = gt_mask,
		predicted_mask = pred_mask,
		predicted_building_heatmap = pred_building_heatmap
	)

# 데스트 결과 측정
test_epoch = smp.utils.train.ValidEpoch(
	model,
	loss=criterion,
	metrics=metrics,
	device=device,
	verbose=True
)

valid_logs = test_epoch.run(test_dataloader)

print('evaluation os test data: ')
print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")

# 결과 DataFrame 변환
train_logs_df = pd.DataFrame(train_logs_list)
valid_logs_df = pd.DataFrame(valid_logs_list)

# 학습 그래프 시각화
plt.figure(figsize=(20, 8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label='Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label="Valid")

plt.xlabel('Epochs', fontsize=21)
plt.ylabel('IoU Score', fontsize=21)
plt.title('IoU Score Plot', fontsize=21)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig('iou_score_plt.png')
plt.show()

plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label = 'Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label = 'Valid')
plt.xlabel('Epochs', fontsize=21)
plt.ylabel('Dice Loss', fontsize=21)
plt.title('Dice Loss Plot', fontsize=21)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig('dice_loss_plot.png')
plt.show()
