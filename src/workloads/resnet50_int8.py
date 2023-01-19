from torchvision.models.quantization import resnet50

# more info:
# https://pytorch.org/vision/stable/models/generated/torchvision.models.quantization.resnet50.html#torchvision.models.quantization.resnet50
model = resnet50(pretrained=True, quantize=True)
