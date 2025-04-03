import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


def init_weights(m):
    """
    Initialization Weights
    """
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d, nn.BatchNorm2d)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class DropPath(nn.Module):
    """
    Drop Path (To Address Overfitting)
    """
    def __init__(self,
                 drop_prob=0.0):
        super(DropPath, self).__init__()

        self.drop_prob = drop_prob

    def drop_path(self, x, drop_prob: float = 0.0, training: bool = False):

        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1.0 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # B (1,)*(3-1) -> B 1 1
        # 1.0 - drop_prob + [0.0, 1,0) -> [1.0 - drop_prob, 2.0 - drop_prob)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        # 1.0 - drop_prob -> 0,  2.0 - drop_prob -> 1
        random_tensor.floor_()
        # Keep Mathematical Expectations Consistent (without drop_path)
        # (1-p)*a+p*0 = (1-p)a, need div 1-p
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):  # B N C
        # Drop Path
        x = self.drop_path(x, self.drop_prob)  # B N C
        return x


class FeedForwardNet(nn.Module):
    """
    Feed Forward Net (FFN)
    """
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_function=nn.GELU,
                 dropout_ratio=0.0):
        super(FeedForwardNet, self).__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_function()
        self.drop = nn.Dropout(p=dropout_ratio)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbedding(nn.Module):
    """
    Patch Embedding
    """
    def __init__(self,
                 in_channels,
                 patch_size,
                 embedding_dimensions,
                 norm_layer=None):
        super(PatchEmbedding, self).__init__()

        # Projection + Normalization
        self.proj = nn.Conv2d(in_channels, embedding_dimensions, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embedding_dimensions) if norm_layer else nn.Identity()

    def forward(self, x):  # B 3 H W
        # Projection + Flatten + Normalization
        x = self.proj(x)  # B C H//P W//P
        x = x.flatten(2).transpose(1, 2)  # B C H//P*W//P (B C N) -> B N C
        x = self.norm(x)  # B N C, if norm_layer=None then f(x)=x
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi Head Self Attention (MHSA)
    """
    def __init__(self,
                 embedding_dimensions,
                 heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_drop_ratio=0.0,
                 projection_drop_ratio=0.0):
        super(MultiHeadSelfAttention, self).__init__()

        # Heads + Head Dimension (C_head) + Scale
        self.heads = heads
        head_dimension = embedding_dimensions // heads
        self.scale = qk_scale or head_dimension ** -0.5
        # QKV Linear
        self.qkv = nn.Linear(embedding_dimensions, embedding_dimensions * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attention_drop_ratio)
        # Output Projection
        self.proj = nn.Linear(embedding_dimensions, embedding_dimensions)
        self.proj_drop = nn.Dropout(p=projection_drop_ratio)

    # query: B N C
    def forward(self, query=None):
        # Calculate Q, K, V
        B, N, C = query.shape
        # B N C -> B N 3C -> B N 3 heads C_head -> 3 B heads N C_head
        qkv = self.qkv(query).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        # Q: B heads N C_head,  K: B heads N C_head,  V: B heads N C_head
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Calculate Q and K Attention
        # B heads N C_head @ B heads C_head N -> B heads N N
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # Attention @ V
        # B heads N N @ B heads N C_head -> B heads N C_head -> B N heads C_head -> B N C
        x = (attn @ v).transpose(1, 2).reshape(B, query.shape[1], C)
        # Output Projection
        x = self.proj(x)
        x = self.proj_drop(x)  # B query.shape[1] C(Decoder output) or B N C(Encoder output)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    """
    def __init__(self,
                 embedding_dimensions,
                 heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_drop_ratio=0.0,
                 projection_drop_ratio=0.0,
                 init_value=1e-4,
                 drop_path_ratio=0.0,
                 norm_layer=nn.LayerNorm,
                 ffn_ratio=4.0,
                 act_function=nn.GELU,
                 dropout_ratio=0.0):
        super(TransformerEncoderLayer, self).__init__()

        # LN + MHSA
        self.norm1 = norm_layer(embedding_dimensions)
        self.attn = MultiHeadSelfAttention(embedding_dimensions, heads, qkv_bias, qk_scale,
                                           attention_drop_ratio, projection_drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()
        # LN + FFN
        self.norm2 = norm_layer(embedding_dimensions)
        ffn_hidden_dimension = int(embedding_dimensions * ffn_ratio)
        self.ffn = FeedForwardNet(in_features=embedding_dimensions, hidden_features=ffn_hidden_dimension,
                                  act_function=act_function, dropout_ratio=dropout_ratio)
        self.gamma_1 = nn.Parameter(init_value * torch.ones((embedding_dimensions)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_value * torch.ones((embedding_dimensions)), requires_grad=True)

    def forward(self, x):  # B N C
        # LN + MHSA + Residual
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))  # B N C
        # LN + FFN + Residual
        x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))  # B N C
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    """
    def __init__(self,
                 depths,
                 embedding_dimensions,
                 heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_drop_ratio=0.0,
                 projection_drop_ratio=0.0,
                 init_value=1e-4,
                 drop_path_ratio=0.0,
                 norm_layer=nn.LayerNorm,
                 ffn_ratio=4.0,
                 act_function=nn.GELU,
                 dropout_ratio=0.0):
        super(TransformerEncoder, self).__init__()

        # drop_path_ratio, stochastic depth decay rule
        drop_path_array = [x.item() for x in torch.linspace(0, drop_path_ratio, depths)]
        # Transformer Encoder
        # Note: nn.Sequential() only support single input and single output, using nn.Sequential
        self.encoder_layers = nn.Sequential(*[
                TransformerEncoderLayer(embedding_dimensions, heads, qkv_bias, qk_scale, attention_drop_ratio,
                                        projection_drop_ratio, drop_path_array[i], init_value, norm_layer,
                                        ffn_ratio, act_function, dropout_ratio)
                for i in range(depths)
            ])
        self.norm = norm_layer(embedding_dimensions)

    def forward(self, x):  # B N C
        # Transformer Encoder
        x = self.encoder_layers(x)  # B N C
        x = self.norm(x)
        return x
        # token = x[:, 0]  # B C
        # return token


class VisionTransformer(nn.Module):
    """
    Vision Transformer
    """
    def __init__(self,
                 input_size=(14, 14),
                 in_channels=768,
                 patch_size=1,
                 embedding_dimensions=768,
                 encoder_depths=12,
                 encoder_heads=12,
                 qkv_bias=True,
                 qk_scale=None,
                 attention_drop_ratio=0.,
                 projection_drop_ratio=0.,
                 init_value=1e-4,
                 drop_path_ratio=0.,
                 norm_layer=nn.LayerNorm,
                 ffn_ratio=4.0,
                 act_function=nn.GELU,
                 dropout_ratio=0.):
        super(VisionTransformer, self).__init__()

        # Patch Embedding & Position Embedding
        # Patch Embedding
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embedding_dimensions)  # B N C
        self.patch_grid_size = (input_size[0] // patch_size, input_size[1] // patch_size)
        patches = self.patch_grid_size[0] * self.patch_grid_size[1]
        # Add Token
        # self.token = nn.Parameter(torch.randn(1, 1, embedding_dimensions))  # 1 1 C, torch.randn
        # Position Embedding
        self.position_embedding = nn.Parameter(torch.zeros(1, patches, embedding_dimensions))  # 1 N C, torch.zeros  # mod
        self.position_drop = nn.Dropout(p=dropout_ratio)

        # ViT Encoder
        self.encoder = TransformerEncoder(encoder_depths, embedding_dimensions, encoder_heads,
                                          qkv_bias, qk_scale, attention_drop_ratio, projection_drop_ratio, init_value,
                                          drop_path_ratio, norm_layer, ffn_ratio, act_function, dropout_ratio)
        # Token & Position Embedding init
        # trunc_normal_(self.token, std=0.02)
        trunc_normal_(self.position_embedding, std=0.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):  # B N C
        # Patch Embedding & Position Embedding
        x = self.patch_embedding(x)  # B N C
        # token = self.token.expand(x.shape[0], -1, -1)  # 1 1 C -> B 1 C
        x = self.position_drop(x + self.position_embedding)  # keep B N C
        # x = torch.cat((token, x), dim=1)  # B N C
        # ViT Encoder
        x = self.encoder(x)  # B N C
        return x


class FAM(nn.Module):
    """
    FAM (Feature Aggregate Moudle)
    """
    def __init__(self,
                 input_size=(56, 56),
                 in_channels=256,
                 patch_size=4,
                 embedding_dimensions=384,
                 encoder_depths=3,
                 encoder_heads=6,
                 drop_path_ratio=0.,
                 dropout_ratio=0.):
        super(FAM, self).__init__()

        # Vision Transformer
        self.encoder = VisionTransformer(input_size, in_channels, patch_size, embedding_dimensions, encoder_depths,
                                     encoder_heads, drop_path_ratio=drop_path_ratio, dropout_ratio=dropout_ratio)
        # Weight init
        # self.apply(init_weights)

    def forward(self, x):  # B 3 H W
        # Encoder
        x = self.encoder(x)  # B C
        B, _, C = x.shape  # B N C
        H, W = self.encoder.patch_grid_size
        x = x.transpose(1, 2).view(B, C, H, W)  # B N C -> B C N -> B C H W
        return x


if __name__ == '__main__':
    pass
    input = torch.rand([2, 256, 56, 56]).cuda()
    net = FAM().cuda().eval()
    output = net(input)
    # print(f"output: {output}")
    print(f"output.shape: {output.shape}")  # torch.Size([2, 384, 14, 14])