import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

dir_data = './dataset'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size
nframe = img_label.n_frames

nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train = os.path.join(dir_data, 'train')
if not os.path.exists(dir_save_train):
  os.makedirs(dir_save_train)
dir_save_val = os.path.join(dir_data, 'val')
if not os.path.exists(dir_save_val):
  os.makedirs(dir_save_val)
dir_save_test = os.path.join(dir_data, 'test')
if not os.path.exists(dir_save_test):
  os.makedirs(dir_save_test)

# 전체 이미지를 섞어줌
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

# train 이미지를 npy 파일로 저장
offset_nframe = 0

for i in range(nframe_train):
  img_label.seek(id_frame[i+offset_nframe])
  img_input.seek(id_frame[i+offset_nframe])
  
  label_ = np.asarray(img_label)
  input_ = np.asarray(img_input)
  
  np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
  np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)
  
# val 이미지를 npy 파일로 저장
offset_nframe = nframe_train

for i in range(nframe_val):
  img_label.seek(id_frame[i+offset_nframe])
  img_input.seek(id_frame[i+offset_nframe])
  
  label_ = np.asarray(img_label)
  input_ = np.asarray(img_input)
  
  np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
  np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), input_)
  
# test 이미지를 npy 파일로 저장
for i in range(nframe_test):
  img_label.seek(id_frame[i+offset_nframe])
  img_input.seek(id_frame[i+offset_nframe])
  
  label_ = np.asarray(img_label)
  input_ = np.asarray(img_input)
  
  np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
  
  plt.subplot(122)
  plt.imshow(label_, cmap='gray')
  plt.title('label')
  
  plt.subplot(121)
  plt.imshow(input_, cmap='gray')
  plt.title('input')
  
  plt.show()