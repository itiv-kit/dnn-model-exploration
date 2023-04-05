from torchvision.models import resnet18, ResNet18_Weights

from model_explorer.accuracy_functions.classification_accuracy import compute_classification_accuracy


# Simply take the ResNet-18 available in torchvision
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

accuracy_function = compute_classification_accuracy
