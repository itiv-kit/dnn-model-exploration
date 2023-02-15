from torchvision import models

# Simply take the ResNet-50 available in torchvision
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
