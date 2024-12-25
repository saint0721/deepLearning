import torch
from torchvision import models
from torch import nn
from transform.model.UNet import UNet
from transform.model.ResNet import ResNet

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    
    if model_name == "ResNet":
        model_ft = ResNet()
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

    elif model_name == 'UNet':
        model_ft = UNet(up_sample_mode = 'conv_transpose')
        set_parameter_requires_grad(model_ft, feature_extract)
        # Last Conv2d layer
        model_ft.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        input_size = 512

    else:
        print('Invalid model name, exiting...')
        exit()

    return model_ft, input_size, model_name
