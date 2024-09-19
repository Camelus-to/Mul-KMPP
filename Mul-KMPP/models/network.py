import torch
import torch.nn as nn
# from models import
from models import shufflenetv2



def make_network(cfg, checkpoint_path, input_3x3=False, pretrained=False, n_channels=1):
    if cfg.backbone_name == 'resnet18':
        net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        if input_3x3:
            net.conv1 = nn.Sequential(
                nn.Conv2d(n_channels, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
        else:
            net.conv1 = nn.Conv2d(n_channels, 64, 7, 2, 3, bias=False)

    elif cfg.backbone_name == 'shufflenetv2':

        net = shufflenetv2.get_model(
            num_classes=cfg.n_pn_classes,
            sample_size=cfg.sample_size,
            width_mult=cfg.width_mult)
        net.conv_last = nn.Sequential(
            nn.Conv3d(net.stage_out_channels[-2], net.stage_out_channels[-1], 3, 1, 0, bias=False),
            nn.BatchNorm3d(net.stage_out_channels[-1]),
            nn.ReLU(inplace=True)
        )
        if n_channels != 3:
            net.conv1[0] = nn.Conv3d(n_channels, 24, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                                     bias=False)
        return net


class CNN3D(nn.Module):
    def __init__(self, n_channel):
        super().__init__()

        self.layer0 = nn.Sequential(
            nn.Conv3d(n_channel, 8, 3, 1, 1, bias=False),
            nn.Conv3d(8, 8, 3, 1, 1, bias=False),
        )
        self.layer1 = nn.Sequential(
            nn.Conv3d(8, 16, 3, 2, 1, bias=False),
            nn.Conv3d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, 2, 1, bias=False),
            nn.Conv3d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 2, 1, bias=False),
            nn.Conv3d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 2, 1, bias=False),  # Customized0
            # nn.Conv3d(128, 128, 3, 1, 1, bias=False),  # Customized0
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
