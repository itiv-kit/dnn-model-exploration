import os
import torch

from model_explorer.third_party.deeplab_v3_pytorch.network.modeling import deeplabv3plus_mobilenet


def deeplabv3plus_mobilenet_init(**kwargs):
    n_classes = kwargs.get('n_classes', 19)

    model = deeplabv3plus_mobilenet(
        num_classes=n_classes,  # 19 for cityscapes
        output_stride=16
    )

    state_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                              "param_checkpoints", "best_deeplabv3plus_mobilenet_cityscapes_os16.pth")
    checkpoint = torch.load(state_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])

    return model


model = deeplabv3plus_mobilenet_init()
