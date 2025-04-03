import torch.nn as nn
import torch


class ResBlock(nn.Module):
    """
    ResBlock (ResNet Block)
    """
    def __init__(self, in_channels, squeeze_factor=4):
        super(ResBlock, self).__init__()

        mid_channels = int(in_channels // squeeze_factor)
        out_channels = in_channels
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)

        return out


class ResBlockGroup(nn.Module):
    """
    ResBlockGroup (ResNet Block Group)
    """
    def __init__(self, in_channels, depths):
        super(ResBlockGroup, self).__init__()

        modules = [ResBlock(in_channels=in_channels) for _ in range(depths)]
        self.group = nn.Sequential(*modules)

    def forward(self, x):
        out = self.group(x)
        return out


class RFEM(nn.Module):
    """
    RFEM (Residual Feature Extraction Module) (Sharpness, Contrast and Naturalness)
    """
    def __init__(self, in_channels=1, out_channels=64, depths=6):
        super(RFEM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group = ResBlockGroup(in_channels=out_channels, depths=depths)

    def forward(self, x):  # B 1 224 224
        x = self.conv1(x)  # B 64 112 112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # B 64 56 56

        x = self.group(x)  # B 64 56 56
        return x


if __name__ == '__main__':
    pass
    # Test
    input = torch.rand([2, 1, 224, 224]).cuda()
    net = RFEM().cuda().eval()
    output = net(input)
    # print(f"output: {output}")
    print(f"output.shape: {output.shape}")  # torch.Size([2, 64, 56, 56])
