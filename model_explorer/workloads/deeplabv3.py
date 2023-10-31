import os
import torch

from model_explorer.third_party.deeplab_v3_pytorch.network.modeling import deeplabv3plus_mobilenet, deeplabv3plus_resnet101

from model_explorer.accuracy_functions.segmentation_accuracy import compute_sematic_segmentation_accuracy


def deeplabv3plus_mobilenet_cityscapes_init():
    model = deeplabv3plus_mobilenet(
        num_classes=19,  # 19 for cityscapes
        output_stride=16,
        pretrained_backbone=False
    )

    state_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                              "param_checkpoints", "best_deeplabv3plus_mobilenet_cityscapes_os16.pth")
    checkpoint = torch.load(state_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])

    return model


def deeplabv3plus_resnet101_cityscapes_init():
    model = deeplabv3plus_resnet101(
        num_classes=19,  # 19 for cityscapes
        output_stride=16,
        pretrained_backbone=False
    )

    state_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                              "param_checkpoints", "best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar")
    checkpoint = torch.load(state_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])

    return model


def deeplabv3plus_mobilenet_cityscapes_accuracy(base_model, dataloader_generator, progress=True, title=""):
    return compute_sematic_segmentation_accuracy(base_model, dataloader_generator, progress, title, n_classes=19)


def deeplabv3plus_resnet101_cityscapes_accuracy(base_model, dataloader_generator, progress=True, title=""):
    return compute_sematic_segmentation_accuracy(base_model, dataloader_generator, progress, title, n_classes=19)


# accuracy_function = deeplabv3plus_mobilenet_cityscapes_accuracy
# model = deeplabv3plus_mobilenet_cityscapes_init()

accuracy_function = deeplabv3plus_mobilenet_cityscapes_accuracy
model = deeplabv3plus_mobilenet_cityscapes_init()
