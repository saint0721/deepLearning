import torch
from torchvision import models
from torch import nn
from ResNet import ResNet50

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # (16, 32, 32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # (32, 16, 16)
        self.pool = nn.MaxPool2d(2, 2)  # (32, 8, 8)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # 수정된 입력 차원 (2048)
        self.fc2 = nn.Linear(128, 3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    
    # if model_name == "ResNet":
    #     model_ft = ResNet50()
    #     set_parameter_requires_grad(model_ft, feature_extract)
    #     num_ftrs = model_ft.linear.in_features
    #     model_ft.linear = nn.Linear(num_ftrs, num_classes)
    #     input_size = 224
    if model_name == "ResNet":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == 'EfficientNet':
        model_ft = models.efficientnet_b7(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "DenseNet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == 'SimpleCNN':
        model_ft = SimpleCNN()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc2.in_features
        model_ft.fc2 = nn.Linear(num_ftrs, num_classes)
        input_size = 32
    
    else:
        print('Invalid model name, exiting...')
        exit()

    return model_ft, input_size
