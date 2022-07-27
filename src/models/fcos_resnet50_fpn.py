import torchvision.models as models

model = models.detection.fcos_resnet50_fpn(pretrained = True)