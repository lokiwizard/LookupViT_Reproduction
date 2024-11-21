import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from Vanilla_ViT import Transformer

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

        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)

        self.fc = nn.Linear(dim, dim)

        # 直接调用torch的多头注意力机制
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.self_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)

        # layer normalization
        self.norm = nn.LayerNorm(dim)

        # ffn
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )


    def forward(self, lookup_patches, compressed_patches):
        # lookup_patches: [B, N, D]
        # compressed_patches: [B, M, D]
        # 1. compressed patch 和 lookup patch 之间做交叉注意力
        identity = compressed_patches

        q = self.norm(self.q_linear(compressed_patches))
        k = self.norm(self.k_linear(lookup_patches))
        v = self.v_linear(lookup_patches)

        compressed_patches, attn_weights = self.cross_attention(query=q, key=k, value=v)
        compressed_patches = compressed_patches + identity
        # attn_weights: [B, M, N]
        # 2. compressed patch 之间做自注意力
        compressed_patches = self.self_attention(query=compressed_patches, key=compressed_patches, value=compressed_patches)[0] + compressed_patches
        compressed_patches = self.fc(self.norm(compressed_patches))

        # 3. 复用第一步的attn_weights，需要转置
        attn_weights = attn_weights.transpose(1, 2)
        lookup_patches = lookup_patches + torch.matmul(attn_weights, compressed_patches)
        lookup_patches = self.norm(lookup_patches)

        lookup_patches = self.ffn(lookup_patches)
        compressed_patches = self.ffn(compressed_patches)

        return lookup_patches, compressed_patches


class LookupViT(nn.Module):

    def __init__(self, in_channels, dim, heads, depth, num_classes, image_size, lookup_patch_size, num_compressed_patches, dropout = 0.1):
        super().__init__()

        assert image_size % lookup_patch_size == 0, 'image size must be divisible by lookup patch size'
        self.num_lookup_patches = (image_size // lookup_patch_size) ** 2
        # compressed patch的数量需要比lookup patch的数量少
        assert (num_compressed_patches ** 2) < self.num_lookup_patches, 'num_compressed_patches must be less than num_lookup_patches'
        self.num_compressed_patches = num_compressed_patches
        self.in_channels = in_channels
        self.to_lookup_patch = Rearrange('b c (nh p1) (nw p2) -> b (p1 p2 c) nh nw', p1=lookup_patch_size, p2=lookup_patch_size, c=in_channels)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_lookup_patches, dim))
        self.pos_embedding_compress = nn.Parameter(
            torch.randn(1, self.num_compressed_patches ** 2, dim))

        self.fc = nn.Linear(lookup_patch_size**2 * in_channels, dim)

        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(LookupTransformerBlock(dim, heads, dropout))

        # cls token for compressed patch
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # small vit
        self.small_vit = Transformer(dim, depth=4, heads=heads, dim_head=64, mlp_dim=dim*2, dropout = 0.)
        # mlp head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # img: [B, C, H, W]
        bacth_size = img.shape[0]

        lookup_patches = self.to_lookup_patch(img)  # [B, P1*P2*C, N, N]
        compressed_patches = nn.functional.interpolate(lookup_patches, size=(self.num_compressed_patches, self.num_compressed_patches),
                                                       mode='bilinear',align_corners=False)  # [B, P1*P2*C, M, M]  bilinear插值

        lookup_patches = rearrange(lookup_patches, 'b d n1 n2 -> b (n1 n2) d')  # [B,N,D]
        compressed_patches = rearrange(compressed_patches, 'b d m1 m2 -> b (m1 m2) d')  # [B,M,D]

        lookup_patches = self.fc(lookup_patches)
        compressed_patches = self.fc(compressed_patches)

        lookup_patches += self.pos_embedding
        compressed_patches += self.pos_embedding_compress

        lookup_patches, compressed_patches = self.dropout(lookup_patches), self.dropout(compressed_patches)

        for transformer in self.transformer:
            lookup_patches, compressed_patches = transformer(lookup_patches, compressed_patches)

        # concatenate cls token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=bacth_size)
        compressed_patches = torch.cat((cls_tokens, compressed_patches), dim=1)

        compressed_patches = self.small_vit(compressed_patches)

        # cls token
        cls_token = compressed_patches[:, 0]


        return self.mlp_head(cls_token)



if __name__ == "__main__":

    model = LookupViT(
        in_channels=3,
        dim=256,
        heads=4,
        depth=3,
        num_classes=10,
        image_size=64,
        lookup_patch_size=16,
        num_compressed_patches=2,
    )

    img = torch.randn(1, 3, 64, 64)
    out = model(img)
    print(out.shape)  # [1, 10]

















