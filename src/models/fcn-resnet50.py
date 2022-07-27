from torchvision import models

model = models.segmentation.fcn_resnet50(pretrained=True)