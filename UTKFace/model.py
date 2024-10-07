import torch
from torchvision import models
from torch import nn
from ResNet import ResNet50

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, task_type, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    
    if model_name == "ResNet":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        # 출력 크기 설정 (나이 예측은 1, 성별 예측은 2)
        if task_type == "regression":
            model_ft.fc = nn.Linear(num_ftrs, 1)  # 나이 예측 (회귀) -> 출력 크기 1
        elif task_type == "classification":
            model_ft.fc = nn.Linear(num_ftrs, 2)  # 성별 예측 (이진 분류) -> 출력 크기 2
        input_size = 200
        
    elif model_name == 'EfficientNet':
        model_ft = models.efficientnet_b7(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        if task_type == "regression":
            model_ft.classifier[1] = nn.Linear(num_ftrs, 1)  # 나이 예측 (회귀) -> 출력 크기 1
        elif task_type == "classification":
            model_ft.classifier[1] = nn.Linear(num_ftrs, 2)  # 성별 예측 (이진 분류) -> 출력 크기 2
        input_size = 200

    elif model_name == "DenseNet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        if task_type == "regression":
            model_ft.classifier = nn.Linear(num_ftrs, 1)  # 나이 예측 (회귀) -> 출력 크기 1
        elif task_type == "classification":
            model_ft.classifier = nn.Linear(num_ftrs, 2)  # 성별 예측 (이진 분류) -> 출력 크기 2
        input_size = 200
        
    elif model_name == 'SimpleCNN':
        model_ft = SimpleCNN()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc2.in_features
        if task_type == "regression":
            model_ft.fc2 = nn.Linear(num_ftrs, 1)  # 나이 예측 (회귀) -> 출력 크기 1
        elif task_type == "classification":
            model_ft.fc2 = nn.Linear(num_ftrs, 2)  # 성별 예측 (이진 분류) -> 출력 크기 2
        input_size = 200
    
    else:
        print('Invalid model name, exiting...')
        exit()

    return model_ft, input_size