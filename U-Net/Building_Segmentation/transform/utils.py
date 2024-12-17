import torch
import torch.nn as nn
import segmentation_models_pytorch.utils as utils
import cv2
import numpy as np

def crop_image(image, target_image_dims=[1500, 1500, 3]):
	target_size = target_image_dims[0]
	image_size = len(image)
	padding = (image_size - target_size) // 2

	return image[
		padding: image_size - padding,
		padding: image_size - padding,
		:
	]

class F1Score(nn.Module):
    def __init__(self, beta=1., eps=1e-7):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.__name__ = 'F1Score'
    
    def forward(self, y_pred, y_true):
        y_pred = torch.round(torch.sigmoid(y_pred))
        tp = (y_true * y_pred).sum(dim=(2, 3))
        fp = ((1 - y_true) * y_pred).sum(dim=(2, 3))
        fn = (y_true * (1 - y_pred)).sum(dim=(2, 3))
        precision = tp / (tp+fp+self.eps)
        recall = tp / (tp+fn+self.eps)
        f1 = (1+self.beta**2) * precision * recall / (self.beta**2 * precision + recall + self.eps)
        
        return f1.mean()


def count_buildings(mask):
    _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=4)
    num_buildings = num_labels - 1

    return num_buildings

