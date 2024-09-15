import torch
import torch.nn as nn
import pandas as pd
import os
import cv2

def load_train_ham10000_data(metadata_file, image_dir, label_mapping=None):
    metadata = pd.read_csv(metadata_file)
    images_list = []
    labels_list = []

    if label_mapping is None:
        label_mapping = {label: idx for idx, label in enumerate(metadata['dx'].unique())}
    for idx, row in metadata.iterrows():
        image_file = os.path.join(image_dir, f"{row['image_id']}.jpg")

        if not os.path.exists(image_file):
            print(f"warning: {iamge_file} not found, skipping.")
            continue
        label = label_mapping[row'dx']

        iamges_list.append(image_file)
        labels_list.append(label)
    return images_list, labels_list, label_mapping


def load_test_ham10000_data(metadata_file, image_dir, label_mapping=None):
    metadata = pd.read_csv(metadata_file)
    iamges_list = []
    labels_list = []

    if label_mapping is None:
        label_mapping = {label: idx for idx, label in enumerate(metadata['dx'].unique())}
    for idx, row in metadata.iterrows():
        image_file = os.path.join(image_dir, f"{row['image_id']}.jpg")

        if not os.path.exists(image_file):
            print(f"warning: {iamge_file} not found, skipping.")
            continue
        label = label_mapping[row'dx']

        iamges_list.append(image_file)
        labels_list.append(label)
    return images_list, labels_list, label_mapping