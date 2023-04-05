import torch
import timm
import os

from torch import nn
import torch.nn.functional as F

from model_explorer.accuracy_functions.segmentation_accuracy import compute_sematic_segmentation_accuracy


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('resnet18', features_only=True, pretrained=True)

    def forward(self, x):
        return self.encoder(x)


class ConvBlock(nn.Sequential):
    def __init__(self, ni, nf) -> None:
        layers = [
            nn.Conv2d(in_channels=ni, out_channels=nf, kernel_size=3, padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=nf),
            nn.ReLU()
        ]
        super().__init__(*layers)


class UnetBlock(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super().__init__()

        def noop(x):
            return x

        self.conv1 = ConvBlock(in_channels, channels)
        self.conv2 = ConvBlock(channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def calc_hyperfeats(d1, d2, d3, d4, d5):
    hyperfeats = torch.cat((
        d1,
        F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False),
        F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False),
        F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False),
        F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False)), 1)
    return hyperfeats


class UnetDecoder(nn.Module):
    def __init__(self, fs=32, expansion=4, n_out=1, hypercol=False):
        super().__init__()

        center_ch = 512*expansion
        decoder5_ch = center_ch + (256*expansion)
        channels = 512

        self.hypercol = hypercol
        self.center = nn.Sequential(ConvBlock(center_ch, center_ch),
                                    ConvBlock(center_ch, center_ch//2))
        self.decoder5 = UnetBlock(decoder5_ch, channels, fs)
        self.decoder4 = UnetBlock(256*expansion+fs, 256, fs)
        self.decoder3 = UnetBlock(128*expansion+fs, 128, fs)
        self.decoder2 = UnetBlock(64*expansion+fs, 64, fs)
        self.decoder1 = UnetBlock(fs, fs, fs)
        if hypercol:
            self.logit = nn.Sequential(ConvBlock(fs*5, fs*2),
                                       ConvBlock(fs*2, fs),
                                       nn.Conv2d(fs, n_out, kernel_size=1))
        else:
            self.logit = nn.Sequential(ConvBlock(fs, fs//2),
                                       ConvBlock(fs//2, fs//2),
                                       nn.Conv2d(fs//2, n_out, kernel_size=1))

    def forward(self, feats):
        e1, e2, e3, e4, e5 = feats

        f = self.center(e5)
        d5 = self.decoder5(torch.cat([f, e5], 1))
        d4 = self.decoder4(torch.cat([d5, e4], 1))
        d3 = self.decoder3(torch.cat([d4, e3], 1))
        d2 = self.decoder2(torch.cat([d3, e2], 1))
        d1 = self.decoder1(d2)
        return self.logit(calc_hyperfeats(d1, d2, d3, d4, d5)) if self.hypercol else self.logit(d1)


class UNet(nn.Module):
    def __init__(self, fs=32, expansion=4, n_out=1, hypercol=False):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = UnetDecoder(fs=fs, expansion=expansion, n_out=n_out, hypercol=hypercol)

    def forward(self, x):
        feats = self.encoder(x)  # '64 256 512 1024 2048'
        out = self.decoder(feats)
        return out


def init_unet_model():
    model = UNet(fs=32, expansion=1, n_out=13)
    fn = os.path.join(os.path.dirname(__file__), 'param_checkpoints', 'resnet18_unet.pth')
    state_dict = torch.load(fn)
    model.load_state_dict(state_dict)

    return model


def unet_david_accuracy(base_model, dataloader_generator, progress=True, title=""):
    return compute_sematic_segmentation_accuracy(base_model, dataloader_generator, progress,
                                                 title, n_classes=13, crop_range=[4, -4])


accuracy_function = unet_david_accuracy
model = init_unet_model()
