import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from utils import F1Score
from visualize import colour_code_segmentation, visualize

def train_and_evaluate(models, epochs, train_loader, valid_loader, test_loader, device, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory '{save_path}' created for saving models")

    train_logs_list = {name: [] for name in models.keys()}
    valid_logs_list = {name: [] for name in models.keys()}

    for name, model in models.items():
        print(f"\nTraining {name}")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        loss = smp.utils.losses.DiceLoss()
        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
            smp.utils.metrics.Precision(threshold=0.5),
            smp.utils.metrics.Recall(threshold=0.5),
            smp.utils.metrics.Accuracy(threshold=0.5),
            F1Score(beta=1, eps=1e-7),
        ]

        train_epoch = smp.utils.train.TrainEpoch(
            model, loss=loss, metrics=metrics, optimizer=optimizer, device=device, verbose=True
        )
        valid_epoch = smp.utils.train.ValidEpoch(
            model, loss=loss, metrics=metrics, device=device, verbose=True
        )
        
        best_iou_score = 0.0
        for epoch in range(epochs):
            print(f"\n{name} - Epoch {epoch + 1}")
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            
            train_logs_list[name].append(train_logs)
            valid_logs_list[name].append(valid_logs)

            if valid_logs['iou_score'] > best_iou_score:
                best_iou_score = valid_logs['iou_score']
                torch.save(model.state_dict(), f"{save_path}/{name}_best_model.pth")
                print(f"{name} model saved with IoU: {best_iou_score:.4f}")

    return train_logs_list, valid_logs_list

def predict_and_visualize(models, test_loader, test_dataset_vis, device, select_classes, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Directory '{output_folder}' created for saving predictions")
    
    for name, model in models.items():
        print(f"\nPredicting with {name}...")
        model = model.to(device)
        model.eval()
        
        for idx, (image, gt_mask) in enumerate(test_loader):
            try:
                # `image_vis`와 `gt_mask`를 numpy로 변환
                image_vis = test_dataset_vis[idx][0].astype('uint8')
                gt_mask = gt_mask.squeeze().cpu().numpy().astype('uint8')

                x_tensor = image.to(device)  # 배치 입력 텐서
                
                with torch.no_grad():
                    pred_logits = model(x_tensor)  # 모델 예측
                
                pred_mask = torch.argmax(pred_logits, dim=1).squeeze().cpu().numpy().astype('uint8')

                # 색상 코딩
                pred_colored = colour_code_segmentation(pred_mask, select_classes)
                gt_colored = colour_code_segmentation(gt_mask, select_classes)
                
                # 크기 조정
                target_size = (image_vis.shape[1], image_vis.shape[0])  # (가로, 세로)
                pred_colored_resized = cv2.resize(pred_colored, target_size, interpolation=cv2.INTER_NEAREST)
                gt_colored_resized = cv2.resize(gt_colored, target_size, interpolation=cv2.INTER_NEAREST)
                
                # 이미지 저장 및 시각화
                output_path = os.path.join(output_folder, f"{name}_prediction_{idx}.png")
                combined_image = np.hstack([
                    image_vis.transpose(1, 2, 0),  # (채널, 높이, 너비) → (높이, 너비, 채널)
                    gt_colored_resized,
                    pred_colored_resized
                ])
                cv2.imwrite(output_path, combined_image[:, :, ::-1])  # OpenCV는 BGR 형식
                visualize(
                    original_image=image_vis.transpose(1, 2, 0),
                    ground_truth_mask=gt_colored_resized,
                    predicted_mask=pred_colored_resized
                )
            except Exception as e:
                print(f"Error at index {idx}: {e}")