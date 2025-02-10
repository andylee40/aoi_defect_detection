import torchvision.models as models
import torch
import torch.nn as nn


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.image_model=models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.num_classes = 6
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, self.num_classes)
    def forward(self, image):
        prob_vec = self.image_model(image)
        prob_softmax = torch.softmax(prob_vec, dim=1)
        pred_labels = torch.argmax(prob_softmax, dim=1)
        return prob_vec,pred_labels
    
class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.image_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.num_classes = 6
        in_features = self.image_model.classifier.in_features
        self.image_model.classifier = nn.Linear(in_features, self.num_classes)

    def forward(self, image):
        prob_vec = self.image_model(image)  # DenseNet的輸出
        prob_softmax = torch.softmax(prob_vec, dim=1)
        pred_labels = torch.argmax(prob_softmax, dim=1)
        return prob_vec, pred_labels
    
class EfficientNetB0(nn.Module):
    def __init__(self):
        super(EfficientNetB0, self).__init__()
        self.image_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.num_classes = 6
        in_features = self.image_model.classifier[1].in_features
        self.image_model.classifier[1] = nn.Linear(in_features, self.num_classes)

    def forward(self, image):
        prob_vec = self.image_model(image)  
        prob_softmax = torch.softmax(prob_vec, dim=1)
        pred_labels = torch.argmax(prob_softmax, dim=1)
        return prob_vec, pred_labels
    
# model = Resnet50()
# model.to(device)