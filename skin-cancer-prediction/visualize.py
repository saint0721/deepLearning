import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import itertools, random

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation='45')
  plt.yticks(tick_marks, classes)

  if normalize:
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  thresh = cm.max() / 2

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape(1))):
    plt.text(j, i, cm[i, j], horizontalalignment='center',
    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.xlabel("Predict label")
    plt.ylabel("True label")


def view_sample(imageid_path_dict, all_image_path):
  fig = plt.figure(figsize=(15, 15))
  columns, rows = 3, 2
  start, end = 0, len(imageid_path_dict)
  ax = []
  for i in range(columns*rows):
    k = random.randint(start, end)
    img = mping/imread((all_image_path[k]))
    ax.append(fig.add_subplot(rows, columns, i+1))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='gray')
  plt.tight_layout()
  plt.show()