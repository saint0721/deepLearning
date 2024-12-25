import pandas as pd
import numpy as np

def get_class_label(file_path):
  class_dict = pd.read_csv(file_path)

  # 클래스 이름
  class_names = class_dict['name'].tolist()

  # 클래스의 RGB값
  class_rgb_values = class_dict[['r','g','b']].values.tolist()

  # 결과 확인용 print()
  print('All dataset classes and their corresponding RGB values in labels:')
  print('Class Names: ', class_names)
  print('Class RGB values: ', class_rgb_values)

  # class name 불러오기
  select_classes = ['background', 'building']

  # RGB 값 불러오기
  select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
  select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

  print('Selected classes and their corresponding RGB values in labels:')
  print('Class Names: ', select_class_indices)
  print('Class RGB values: ', select_class_rgb_values)
  
  return select_class_rgb_values, select_classes

