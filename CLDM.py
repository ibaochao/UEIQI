import torch
import torch.nn as nn


class FeatureExtraction(nn.Module):
    """
    Feature Extraction (FE)
    """

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=4):
        super(FeatureExtraction, self).__init__()

        # FE
        self.fe = nn.Sequential(
            # Conv + IN + ReLU
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size // 2), bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # B 3 H W
        # FE
        x = self.fe(x)  # B C H//2 W//2
        return x


class CA(nn.Module):
    """
    CA (Channel Attention)
    """
    def __init__(self, in_channels, ratio=16):
        super(CA, self).__init__()

        # Avg Pool & Max pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # shared_MLP
        self.ca = nn.Sequential(
            # Conv + ReLU + Conv (Replace Linear + ReLU + Linear)
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        # Sigmoid -> Weight(C*1*1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # B C H W
        # CA -> Weight
        w = self.ca(self.avg_pool(x)) + self.ca(self.max_pool(x))
        w = self.sigmoid(w)  # B C 1 1
        # Weight * X
        x = w * x  # B C H W
        return x


class SA(nn.Module):
    """
    SA (Spatial Attention)
    """
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Sigmoid -> Weight(1*H*W)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # B C H W
        # SA -> Weight
        avg_out = torch.mean(x, dim=1, keepdim=True)  # B 1 H W
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # B 1 H W
        w = self.conv(torch.cat([avg_out, max_out], dim=1))  # B 2 H W -> B 1 H W
        # Weight * X
        x = w * x  # B C H W
        return x


class CBAM(nn.Module):
    """
    CBAM (Convolutional Block Attention Module)
    """
    def __init__(self, in_channels):
        super(CBAM, self).__init__()

        # CA
        self.ca = CA(in_channels)
        # SA
        self.sa = SA()


    def forward(self, x):  # B C H W
        # CA
        x = self.ca(x)  # B C H W
        # SA
        x = self.sa(x)  # B C H W
        return x


class CBAMGroup(nn.Module):
    """
    CBAMGroup (CBAM Group)
    """
    def __init__(self, in_channels, depths):
        super(CBAMGroup, self).__init__()

        modules = [CBAM(in_channels=in_channels) for _ in range(depths)]
        self.group = nn.Sequential(*modules)

    def forward(self, x):
        out = self.group(x)
        return out


class CLDM(nn.Module):
    """
    CLDM (Channel and Local Distortions Module)
    """
    def __init__(self, in_channels=3, dims=64, depths=3):
        super(CLDM, self).__init__()

        # FE
        self.fe = FeatureExtraction(in_channels=in_channels, out_channels=dims)
        # CBAMs
        self.cbams = CBAMGroup(in_channels=dims, depths=depths)


    def forward(self, x):  # B 3 224 224
        # FE
        x = self.fe(x)  # B 64 56 56
        # CBAMs
        x = self.cbams(x)  # B 64 56 56
        return x


if __name__ == "__main__":
    pass
    # Test
    input = torch.rand([2, 3, 224, 224]).cuda()
    net = CLDM().cuda().eval()
    output = net(input)
    # print(f"output: {output}")
    print(f"output.shape: {output.shape}")  # torch.Size([2, 64, 56, 56])