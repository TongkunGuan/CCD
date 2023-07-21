# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from Dino.modules.vision_transformer import Block


class SEGDecoder(nn.Module):
    def __init__(self,
                 num_patches=256, patch_size=4, num_classes=1, embed_dim=384, depth=3,
                 num_heads=2, mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer)
            for i in range(depth)])
        self.decoder_norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size ** 2 * num_classes, bias=True)

    def forward(self, x):
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        outputs = self.decoder_pred(x)
        return outputs
