import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from CLDM import CLDM  # Channel and Local Distortions Module
from RFEM import RFEM  # Sharpness and Contrast & Naturalness
from FDEM import FDEM  # Fog Density Estimation Module
from FAM import FAM  # Feature Aggregate Moudle



class QMM(nn.Module):
    """
    QMM (Quality Mapping Moudle)
    """
    def __init__(self, in_channels, dims):
        super(QMM, self).__init__()
        # DS
        self.ds = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.map = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_channels * 2, dims),
        )

    def forward(self, x):  # B C H W
        x = self.ds(x)  # B 2C(768) H//2(7) W//2(7)
        x = self.map(x)  # B dims
        return x


class UEIQI(nn.Module):
    """
    UEIQI (Underwater Enhanced Image Quality Index)
    """
    def __init__(self,
                 # -------Channel and Local Distortions Module
                 cldm_in_channels=3,
                 cldm_dims=64,
                 cldm_depths=3,
                 # -------Sharpness and Contrast & Naturalness
                 rfem_in_channels=1,
                 rfem_out_channels=64,
                 rfem_depths=6,
                 # -------Fog Density Estimation Module
                 fdem_in_channels=3,
                 fdem_embed_dim=64,
                 fdem_img_size=224,
                 fdem_patch_size=4,
                 fdem_depths=1,
                 fdem_drop=0.1,
                 fdem_drop_path=0.1,
                 # -------Feature Aggregate Moudle
                 fam_input_size=(56, 56),
                 fam_in_channels=256,
                 fam_patch_size=4,
                 fam_embedding_dimensions=384,
                 fam_encoder_depths=3,
                 fam_encoder_heads=6,
                 fam_drop_path_ratio=0.,
                 fam_dropout_ratio=0.):
        super(UEIQI, self).__init__()

        # CLDM
        self.cldm = CLDM(in_channels=cldm_in_channels, dims=cldm_dims, depths=cldm_depths)
        # RFEM1 (GM)
        self.rfem1 = RFEM(in_channels=rfem_in_channels, out_channels=rfem_out_channels, depths=rfem_depths)
        # RFEM2 (MSCN)
        self.rfem2 = RFEM(in_channels=rfem_in_channels, out_channels=rfem_out_channels, depths=rfem_depths)
        # FDEM
        self.fdem = FDEM(in_channels=fdem_in_channels, embed_dim=fdem_embed_dim, img_size=fdem_img_size, patch_size=fdem_patch_size, 
                         depths=fdem_depths, drop=fdem_drop, drop_path=fdem_drop_path)
        # FAM
        self.fam = FAM(input_size=fam_input_size, in_channels=fam_in_channels, patch_size=fam_patch_size,
                       embedding_dimensions=fam_embedding_dimensions, encoder_depths=fam_encoder_depths,
                       encoder_heads=fam_encoder_heads, drop_path_ratio=fam_drop_path_ratio,
                       dropout_ratio=fam_dropout_ratio)
        # QMM
        self.qmm = QMM(in_channels=fam_embedding_dimensions, dims=1)


    def forward(self, x, gm, mscn):  # x: B 3 H W, gm & mscn: B 1 H W
        # CLDM
        f_cldm = self.cldm(x)  # B 64 56 56
        # RFEM1 (GM)
        f_rfem_gm = self.rfem1(gm)  # B 64 56 56
        # RFEM2 (MSCN)
        f_rfem_mscn = self.rfem2(mscn)  # B 64 56 56
        # FDEM
        f_fdem = self.fdem(x)  # B 64 56 56
        # Concat
        f_4 = torch.cat([f_cldm, f_rfem_gm, f_rfem_mscn, f_fdem], dim=1)  # B 256
        # FAM
        f = self.fam(f_4)  # B 384 14 14
        # QMM
        qualityscore = self.qmm(f)  # B 1
        return qualityscore


if __name__ == "__main__":
    pass
    # Test
    input = torch.rand([2, 3, 224, 224]).cuda()
    input2 = torch.rand([2, 1, 224, 224]).cuda()
    input3 = torch.rand([2, 1, 224, 224]).cuda()
    net = UEIQI().cuda().eval()
    output = net(input, input2, input3)
    # print(f"output: {output}")
    print(f"output.shape: {output.shape}")  # torch.Size([2, 1])



