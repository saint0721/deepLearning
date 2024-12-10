# 파이썬 라이브러리
import pandas as pd
import numpy as np
import os, cv2
from tqdm import tqdm

def compute_mean_stdev(image_paths):
    img_h, img_w = 200, 200
    imgs = []
    means, stdevs = [], []

    # tqdm: 진행도 확인할 수 있는 라이브러리ㅣ
    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i]) # BGR로 읽음
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)
    
    # (N, 224, 224, 3)으로 변환 axis=0 행, axis=1로 하면 열로 만들어짐
    imgs = np.stack(imgs, axis=0) 
    print(imgs.shape) # 이미지 객체 배열 확인
    imgs = imgs.astype(np.float32) / 255. # 픽셀값 정규화

    for i in range(3): # BGR 채널별로 계산
        pixels = imgs[:, :, :, i].ravel() # 1차원 배열로 변환
        means.append(np.mean(pixels)) # 평균 계산
        stdevs.append(np.std(pixels)) # 표준편차 계산
    
    means.reverse() # RGB로 변환
    stdevs.reverse() # RGB로 변환

    print(f"NormMean: {means}")
    print(f"normStdev: {stdevs}")
    return means, stdevs
        
class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

cam_lambda = 0.2
save_dir = os.path.join(f"./saved_sample")

def model_to_cam(model, img_size, norm_stdevs):
    model = self.model
    img_size = self.img_size
    norm_stdevs = self.norm_stdevs

    for class_idx in range(117):
        os.makedirs(os.path.join(save_dir, str(class_idx)), exists_ok = True)

    for batch_idx, (data, target) in enumerate(tdqm(train_loader)):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
    
        out = model(data)
        activation_map = localize_net(out.squeeze(0).argmax().item(), out)
        activation_map = activation_map[0].squeeze().detach().cpu().numpy()

        if activation_map.shape[0] != img_size:
            x = cv2.resize(activation_map, (img_size, img_size))
        else:
            x = activation_map
        activation_map = x

        x_data_array = np.transpose(data.detach().cpu().numpy(), [0, 2, 3, 1])
        origin_x_data = (x_data_array * np.array(norm_stdevs).reshape()[1, 1, 1, 3]) + np.array(mean).reshape([1, 1, 1, 3])
        origin_x_data = np.uint8(origin_x_data * 255)[0]

        background_mask = np.uint8(activation_map < cam_lambda)
        remove_image = np.copy(origin_x_data) * np.expand_dims(background_mask, axis=1)

        target_mask = -1 * (background_mask.astype(np.float32) - 1.)
        inpaint = cv2.inpaint(remove_image, target_mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)

        class_idx = target.detach().cpu().numpy().flatten()[0]

        save_original_train_path = os.path.join(save_dir, str(class_idx), f"{batch_idx}_{class_idx}_{str(cam_lambda)}_original.png")
        save_ood_train_path = os.path.join(save_dir, str(class_idx), f"{batch_idx}_{class_idx}_{str(cam_lambda)}_train.png")
        save_ood_mask_path = os.path.join(save_dir, str(class_idx), f"{batch_idx}_{class_idx}_{str(cam_lambda)}_mask.png")

        cv2.imwrite(save_original_train_path, origin_x_data[..., ::-1].astype(np.uint8))
        cv2.imwrite(save_ood_mask_path, (target_mask * 255).astype(np.uint8))
        cv2.imwrite(save_ood_train_path, inpaint[..., ::-1].astype(np.uint8))