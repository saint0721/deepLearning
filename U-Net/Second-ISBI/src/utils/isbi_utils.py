import logging
import numpy as np
import tifffile as tiff

from src.utils.runtime import funcname

def isbi_get_data_montage(imgs_path, msks_path, nb_rows, nb_cols, rng):
  logger = logging.getLogger(funcname())

  imgs, msks = tiff.imread(imgs_path), tiff.imread(msks_path) / 255
  montage_imgs = np.empty((nb_rows * imgs.shape[1], nb_cols * imgs.shape[2]), dtype=np.float32)
  montage_msks = np.empty((nb_rows * imgs.shape[1], nb_cols * imgs.shape[2]), dtype=np.int8)
  # tiff는 한 파일에 여러 이미지가 압축되어 있는 형태(N, H, W), 채널을 포함하면 (N, C, H, W)(Tensor)
  # imgs.shape[0]은 N, imgs.shape[1] H, imgs.shape[2]는 W

  idxs = np.arange(imgs.shape[0])
  rng.shuffle(idxs)
  idxs = iter(idxs)

  for y0 in range(0, montage_imgs.shape[0], imgs.shape[1]):
    for x0 in range(0, montage_imgs.shape[1]. imgs.shape[2]):
      y1, x1 = y0 + imgs.shape[1], x0 + imgs.shape[2]
      idx = next(idxs) # next(): 다음 인덱스 반환
      montage_imgs[y0: y1, x0: x1] = imgs[idx]
      montage_msks[y0: y1, x0: x1] = msks[idx]

  return montage_imgs, montage_msks