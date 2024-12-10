import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO

input_folder = '/home/students/cs/202121165/public_html/deepLearning/cormorant/unlabeled'
output_folder_base = '/home/students/cs/202121165/public_html/deepLearning/cormorant'
visualized_output_folder_base = '/home/students/cs/202121165/public_html/deepLearning/cormorant'

model = YOLO('yolov8x.pt')

class_colors = {
    0: (0, 255, 0),       
    1: (255, 0, 0),       
    2: (0, 0, 255),       
    3: (255, 255, 0),     
    4: (255, 0, 255),     
    5: (0, 255, 255),     
    6: (128, 0, 128),     
    7: (128, 128, 0),     
    8: (128, 128, 128)     
}   


def create_unique_folder(base_path):
    if not os.path.exists(base_path):   
        os.makedirs(base_path)
        return base_path
    else:                                   
        i = 1
        while True:
            new_path = f"{base_path}_{i}"
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                return new_path
            i += 1

def process_and_label_image(image_path, conf_threshold=0.1):
    img = cv2.imread(image_path)  
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    results = model(img_rgb)
    
    filtered_results = results[0].boxes.data.cpu().numpy()
    filtered_results = filtered_results[filtered_results[:, 4] >= conf_threshold]
    
    height, width = img.shape[:2]
    labels = []
    for detection in filtered_results:
        # class_id = int(detection[5])
        class_id = 6
        x_center = (detection[0] + detection[2]) / (2 * width)
        y_center = (detection[1] + detection[3]) / (2 * height)
        w = (detection[2] - detection[0]) / width
        h = (detection[3] - detection[1]) / height
        labels.append(f"{class_id} {x_center} {y_center} {w} {h}")
    
    return labels, img  

def save_labels(labels, output_path):
    with open(output_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

def visualize_and_save_labels(img, labels, output_image_path):
    height, width = img.shape[:2]
    for label in labels:
        class_id, x_center, y_center, w, h = map(float, label.split())
        x_min = int((x_center - w/2) * width)
        y_min = int((y_center - h/2) * height)
        x_max = int((x_center + w/2) * width)
        y_max = int((y_center + h/2) * height)
        
        color = class_colors.get(int(class_id), (255, 255, 255))  

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(img, f"Class {int(class_id)}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2, cv2.LINE_AA)
    cv2.imwrite(output_image_path, img)


output_folder = create_unique_folder(output_folder_base)
visualized_output_folder = create_unique_folder(visualized_output_folder_base)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for i, image_file in enumerate(image_files):
    image_path = os.path.join(input_folder, image_file)
    output_label_path = os.path.join(output_folder, f"{Path(image_file).stem}.txt")
    output_image_path = os.path.join(visualized_output_folder, f"{Path(image_file).stem}_labeled.jpg")
    
    labels, img = process_and_label_image(image_path)
    
    save_labels(labels, output_label_path)
    
    visualize_and_save_labels(img, labels, output_image_path)
    
    print(f"Processed and saved labeled image for: {image_file}")

print("라벨링 완료")