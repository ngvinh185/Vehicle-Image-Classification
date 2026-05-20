import torch.nn as nn
from config import num_class
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import torch
class Model(nn.Module):
  def __init__(self, num_class):
    super().__init__()
    self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1).features
    self.head = nn.ModuleDict(dict(
      custom_conv = nn.Sequential(
        nn.Conv2d(in_channels=1536, out_channels=2048, kernel_size=7, stride=2, padding=1),
        nn.BatchNorm2d(2048),
        nn.ReLU()
      ), 
      fc = nn.Sequential(
        nn.Linear(2048 * 2 * 2, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5), 
        
        nn.Linear(512, num_class)
      )
    ))
  def forward(self, x):
    x = self.backbone(x)
    x = self.head['custom_conv'](x)
    x = x.flatten(1)
    x = self.head['fc'](x)
    return x


    