import torch
from torch import nn
import torch.nn.functional as F
import math


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    drop_path
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    DropPath
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()

        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class PatchEmbedding(nn.Module):
    """
    PatchEmbedding
    """
    def __init__(self, in_chans=3, embed_dim=384, img_size=224, patch_size=16, flatten=True):
        super(PatchEmbedding, self).__init__()

        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B C H W -> B N C
            x = self.norm(x)
        else:
            x = self.norm2(x)
        return x


class FFT(nn.Module):
    """
    FFT (Fast Fourier Transform)
    """
    def __init__(self, dim, h=14, w=8):
        super(FFT, self).__init__()

        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        # self.ca = CA(in_channels=dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.view(B, H, W, C)
        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')  # torch.Size([2, 14, 8, 384])
        weight = torch.view_as_complex(self.complex_weight)  # torch.Size([14, 8, 384])
        # weight = torch.view_as_complex(self.ca(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        # Input type (CUDAComplexFloatType) and weight type (torch.cuda.FloatTensor) should be the same
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)
        return x


class Mlp(nn.Module):
    """
    Mlp
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FEM(nn.Module):
    """
    FEM (Feature Enhancement Module)
    """
    def __init__(self, p=[2, 1, 0]):
        super(FEM, self).__init__()
        self.p = p
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        s1 = int(h / 2 ** self.p[0])  # 4 * 4
        attn1 = torch.zeros(size=(b, c, h, w)).cuda()
        for i in range(0, h, s1):
            for j in range(0, w, s1):
                attn1[:, :, i:i + s1, j:j + s1] = self.sigmoid(F.instance_norm(x[:, :, i:i + s1, j:j + s1]))
        out1 = 0.5 * x + 0.5 * (attn1 * x)

        s2 = int(h / 2 ** self.p[1])  # 2 * 2
        attn2 = torch.zeros(size=(b, c, h, w)).cuda()
        for i in range(0, h, s2):
            for j in range(0, w, s2):
                attn2[:, :, i:i + s2, j:j + s2] = self.sigmoid(F.instance_norm(out1[:, :, i:i + s2, j:j + s2]))
        out2 = 0.5 * out1 + 0.5 * (attn2 * out1)

        s3 = int(h / 2 ** self.p[2])  # 1 * 1
        attn3 = torch.zeros(size=(b, c, h, w)).cuda()
        for i in range(0, h, s3):
            for j in range(0, w, s3):
                attn3[:, :, i:i + s3, j:j + s3] = self.sigmoid(F.instance_norm(out2[:, :, i:i + s3, j:j + s3]))
        final_out = 0.5 * out2 + 0.5 * (attn3 * out2)

        return final_out


class FFTBlock(nn.Module):
    """
    FFTBlock
    """
    def __init__(self, dim, h=14, w=8, drop=0., drop_path=0., norm_layer=nn.LayerNorm, mlp_ratio=4., act_layer=nn.GELU):
        super(FFTBlock, self).__init__()

        self.norm1 = norm_layer(dim)
        self.fft = FFT(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.fem = FEM(p=[2, 1, 0])
        self.h = h
        self.w = self.h

    def forward(self, x):
        x_t = x
        B, _, C = x.shape  # B N C
        x = self.mlp(self.norm2(self.fft(self.norm1(x))))  # B N C
        x = x.transpose(1, 2).view(B, C, self.h, self.w)  # B C H W
        x = self.fem(x)  # B C H W
        x = x.flatten(2).transpose(1, 2)  # B N C
        x = x_t + self.drop_path(x)
        return x


class FDEM(nn.Module):
    """
    FDEM (Fog Density Estimation Module)
    """
    def __init__(self,
                 in_channels=3,
                 embed_dim=64,
                 img_size=224,
                 patch_size=4,
                 depths=1,
                 drop=0.1,
                 drop_path=0.1):
        super(FDEM, self).__init__()

        self.patch_embed = PatchEmbedding(in_chans=in_channels, embed_dim=embed_dim, img_size=img_size, patch_size=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop)

        h = img_size // patch_size
        w = h // 2 + 1
        self.blocks = nn.ModuleList()
        for _ in range(depths):
            layer = FFTBlock(dim=embed_dim, h=h, w=w, drop_path=drop_path)
            self.blocks.append(layer)


    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        B, _, C = x.shape  # B N C
        H = W = self.patch_embed.grid_size
        x = x.transpose(1, 2).view(B, C, H, W)  # B N C -> B C N -> B C H W
        return x


if __name__ == "__main__":
    pass
    # Test
    input = torch.rand([2, 3, 224, 224]).cuda()
    net = FDEM().cuda().eval()
    output = net(input)
    # print(f"output: {output}")
    print(f"output.shape: {output.shape}")  # torch.Size([2, 64, 56, 56])