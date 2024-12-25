import matplotlib.pyplot as plt
import numpy as np

def visualize(**images):
	n_images = len(images)
	plt.figure(figsize=(20, 8))

	for idx, (name, image) in enumerate(images.items()):
		plt.subplot(1, n_images, idx+1)
		plt.xticks([])
		plt.yticks([])
		plt.title(name.replace('_', ' ').title(), fontsize=20)
		plt.imshow(image)
	plt.show()

def one_hot_encode(label, label_values):
	semantic_map = []
	for colour in label_values:
		equality = np.equal(label, colour)
		class_map = np.all(equality, axis= -1)
		semantic_map.append(class_map)
	semantic_map = np.stack(semantic_map, axis= -1)

	return semantic_map

def reverse_one_hot(image):
	x = np.argmax(image, axis= -1)

	return x

def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    max_index = len(colour_codes) - 1  # 허용되는 최대 인덱스
    image = np.clip(image, 0, max_index)  # 인덱스가 범위를 초과하지 않도록 클리핑
    x = colour_codes[image.astype(int)]
    return x

def plot_metrics(model_name, train_logs, valid_logs, metric_names):
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 5*len(metric_names)))

    for i, metric_name in enumerate(metric_names):
        ax = axes[i] if len(metric_name) > 1 else axes
        ax.plot([log[metric_name] for log in train_logs], label=f'{model_name}Train')
        ax.plot([log[metric_name] for log in valid_logs], label=f'{model_name}Validation')
        ax.set_title(f"{metric_name} for {model_name}")
        ax.legend()
    
    plt.tight_layout()
    plt.show()