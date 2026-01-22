import torch
import torch.nn as nn
from torchvision import models

class SimpleSceneCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSceneCNN, self).__init__()
        self.base_model = models.mobilenet_v2(weights='DEFAULT')
        
        # Unfreeze the last 4 layers of the 'eyes' so it can learn sports vs animation textures
        for param in self.base_model.features[-4:].parameters():
            param.requires_grad = True
            
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)