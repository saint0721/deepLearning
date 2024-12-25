import torch
import os, cv2
from visualize import one_hot_encode

class BuildingsDataset(torch.utils.data.Dataset):
	def __init__(self, images_dir, masks_dir, class_rgb_values=None, augmentation=None, preprocessing=None):
   	    # 컴프리헨션 문법으로 이미지 폴더에서 정렬된 image_id추출 
		self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
		# 컴프리헨션 문법으로 마스크 폴더에서 image_id추출
		self.mask_paths = [os.path.join(masks_dir, mask_id) for mask_id in sorted(os.listdir(masks_dir))]

		self.class_rgb_values = class_rgb_values
		self.augmentation = augmentation
		self.preprocessing = preprocessing
  
  
	def __len__(self):
		return len(self.image_paths)
	
	def __getitem__(self, i):
		image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
		mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

		# mask에 one-hot-encode 적용
		mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

		if self.augmentation:
			sample = self.augmentation(image=image, mask=mask)
			image, mask = sample['image'], sample['mask']

		if self.preprocessing:
			sample = self.preprocessing(image=image, mask=mask) 
			image, mask = sample['image'], sample['mask']
		
		return image, mask
