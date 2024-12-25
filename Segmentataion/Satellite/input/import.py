import kagglehub

# Download latest version
path = kagglehub.dataset_download("kmader/synthetic-word-ocr")

print("Path to dataset files:", path)