from torchvision.models import resnet18, ResNet18_Weights

# Simply take the ResNet-18 available in torchvision
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
