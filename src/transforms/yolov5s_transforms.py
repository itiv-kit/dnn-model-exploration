import cv2
import numpy as np
from torchvision.transforms import functional as F

# taken from here: https://github.com/ultralytics/yolov5/blob/5774a1514de193c74ecc5203281da8de3c13f9af/utils/dataloaders.py#L94

class ResizeAndPad:

    def __init__(self, new_shape=(640, 640), color=(114, 114, 114), scaleup=False):

        self.new_shape = new_shape
        if isinstance(self.new_shape, int):
            self.new_shape = (self.new_shape, self.new_shape)

        self.color = color
        self.scaleup = scaleup

    def __call__(self, image, target):

        image = np.array(image)
        # Resize and pad image 
        shape = image.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)  # add border

        return image, target


class ComposeCocoTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

# yolov5 can also take PIL images and apply all needed transforms
transforms = ComposeCocoTransforms([
    # you can add other transformations in this list
    ResizeAndPad(),
    ToTensor(),
])