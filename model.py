"""CSRNet-style CNN: VGG16 front-end (frozen-ish, pretrained) + dilated-conv back-end
that regresses a 1-channel density map at 1/8 resolution. Count = sum of the map."""
import torch.nn as nn
from torchvision import models


def _dilated_backend(in_ch=512):
    cfg = [512, 512, 512, 256, 128, 64]
    layers, c = [], in_ch
    for v in cfg:
        layers += [nn.Conv2d(c, v, 3, padding=2, dilation=2), nn.ReLU(inplace=True)]
        c = v
    return nn.Sequential(*layers)


class CrowdCNN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.VGG16_BN_Weights.DEFAULT if pretrained else None
        vgg = models.vgg16_bn(weights=weights)
        # features[:33] keeps conv1_1 .. conv4_3 (+BN/ReLU) with three max-pools -> 1/8 resolution
        self.frontend = nn.Sequential(*list(vgg.features.children())[:33])
        self.backend = _dilated_backend(512)
        self.output = nn.Conv2d(64, 1, 1)
        self._init_new_layers()

    def _init_new_layers(self):
        for m in list(self.backend.modules()) + [self.output]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output(x)  # (B,1,H/8,W/8); sum over spatial dims = predicted count
