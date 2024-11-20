import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def bilinearResize(x, size):
    # 双线性插值采样
    x = x.transpose(1, 2)
    x = nn.functional.interpolate(x, size=size, mode='linear', align_corners=True)
    x = x.transpose(1, 2)
    return x


class LookupTransformerBlock(nn.Module):
    # 有三个阶段
    # 1. compressed patch 和 lookup patch 之间做交叉注意力
    # 2. compressed patch 之间做自注意力
    # 3. lookup patch 和 compressed patch 之间做交叉注意力
    def __init__(self, dim, heads, dropout = 0.):
        super().__init__()

        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        # 直接调用torch的多头注意力机制
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.self_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)

        # layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # ffn
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )


    def forward(self, lookup_patches, compressed_patches):
        # lookup_patches: [B, N, D]
        # compressed_patches: [B, M, D]
        # 1. compressed patch 和 lookup patch 之间做交叉注意力
        identity = compressed_patches
        compressed_patches, attn_weights = self.cross_attention(query=compressed_patches, key=lookup_patches, value=lookup_patches)
        compressed_patches = compressed_patches + identity
        compressed_patches = self.norm1(compressed_patches)
        # attn_weights: [B, M, N]
        # 2. compressed patch 之间做自注意力
        compressed_patches = self.self_attention(query=compressed_patches, key=compressed_patches, value=compressed_patches)[0] + compressed_patches
        compressed_patches = self.norm2(compressed_patches)

        # 3. 复用第一步的attn_weights，需要转置
        attn_weights = attn_weights.transpose(1, 2)
        lookup_patches = lookup_patches + torch.matmul(attn_weights, compressed_patches)
        lookup_patches = self.norm3(lookup_patches)

        lookup_patches = self.ffn(lookup_patches)
        compressed_patches = self.ffn(compressed_patches)

        return lookup_patches, compressed_patches


class LookupViT(nn.Module):

    def __init__(self, in_channels, dim, heads, depth, num_classes, image_size, lookup_patch_size, dropout = 0.):
        super().__init__()

        assert image_size % lookup_patch_size == 0, 'image size must be divisible by lookup patch size'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=lookup_patch_size, p2=lookup_patch_size),
            nn.Linear(lookup_patch_size * lookup_patch_size * in_channels, dim)
        )

        self.num_lookup_patches = (image_size // lookup_patch_size) ** 2
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_lookup_patches, dim))
        # compressed patch的数量需要比lookup patch的数量少，这里取一半
        self.num_compressed_patches = self.num_lookup_patches // 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(LookupTransformerBlock(dim, heads, dropout))

        # 最后的分类层
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):

        lookup_patches = self.to_patch_embedding(img)
        compressed_patches = bilinearResize(lookup_patches, size=self.num_compressed_patches)
        compressed_pos_embedding = bilinearResize(self.pos_embedding, size=self.num_compressed_patches)

        lookup_patches = lookup_patches + self.pos_embedding
        compressed_patches = compressed_patches + compressed_pos_embedding

        lookup_patches = self.dropout(lookup_patches)
        compressed_patches = self.dropout(compressed_patches)

        # cls token
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=img.shape[0])
        lookup_patches = torch.cat((cls_token, lookup_patches), dim=1)  # [B, N+1, D]
        compressed_patches = torch.cat((cls_token, compressed_patches), dim=1)  # [B, M+1, D]

        for transformer in self.transformer:
            lookup_patches, compressed_patches = transformer(lookup_patches, compressed_patches)

        # 取compress patch的cls token

        return self.mlp_head(compressed_patches[:, 0])

if __name__ == "__main__":

    model = LookupViT(
        in_channels=3,
        dim=256,
        heads=4,
        depth=3,
        num_classes=10,
        image_size=128,
        lookup_patch_size=16
    )

    img = torch.randn(1, 3, 128, 128)
    out = model(img)
    print(out.shape)

















