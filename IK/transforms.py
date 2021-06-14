from torch.nn import Module
from torchvision.transforms.transforms import Resize, Compose
from torchvision.transforms.functional import Image
import cv2

compose = Compose


class Rescale(Module):
    def __init__(self, height=224, width=None, interpolation=cv2.INTER_CUBIC):
        super(Rescale, self).__init__()
        self.interpolation = interpolation
        self.height = height
        self.width = height if width is None else width
        # self.resize = Resize(size=(self.height, self.width), interpolation=interpolation)

    def forward(self, image, *args, **kwargs):
        return cv2.resize(image, dsize=(self.width, self.height), interpolation=self.interpolation)


class LeftMost(Module):
    def __init__(self, width=720):
        super(LeftMost, self).__init__()
        self.width = width

    def forward(self, image, *args, **kwargs):
        return image[:, :self.width]
