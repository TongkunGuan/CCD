import logging
import torch.nn as nn
from fastai.vision import *

from Dino.modules.attention import *
from Dino.modules.backbone import ResTranformer, ResFPNTranformer
from Dino.modules.model import Model
from Dino.modules.resnet import resnet45


class BaseVision(Model):
    def __init__(self, config):
        super().__init__(config)
        self.loss_weight = ifnone(config.model_vision_loss_weight, 1.0)
        self.out_channels = ifnone(config.model_vision_d_model, 512)
        self.config = config
        if config.model_vision_backbone == 'transformer':
            self.backbone = ResTranformer(config)
        elif config.model_vision_backbone == 'transformerFPN':
            self.backbone = ResFPNTranformer(config)
        else:
            self.backbone = resnet45()

        if config.model_contrastive_supervised_flag or config.model_contrastive_kmeans_flag:
            if config.model_vision_attention == 'position':
                mode = ifnone(config.model_vision_attention_mode, 'nearest')
                self.attention = PositionAttention(
                    max_length=config.dataset_max_length + 1,  # additional stop token
                    mode=mode,
                )
            elif config.model_vision_attention == 'attention':
                self.attention = Attention(
                    max_length=config.dataset_max_length + 1,  # additional stop token
                    n_feature=8 * 32,
                )
            else:
                raise NotImplementedError(f'{config.model_vision_attention} is not valid.')
        if config.model_contrastive_supervised_flag:
            self.cls = nn.Linear(self.out_channels, self.charset.num_classes)

        if config.model_vision_checkpoint is not None:
            logging.info(f'Read vision model from {config.model_vision_checkpoint}.')
            self.load(config.model_vision_checkpoint, submodule=config.model_vision_checkpoint_submodule,
                      exclude=config.model_vision_exclude)

    def forward(self, images, mask_flag):
        features, learned_mask = self.backbone(images, mask_flag)  # (N, E, H, W)
        assert self.config.model_contrastive_kmeans_flag != self.config.model_contrastive_supervised_flag

        if self.config.model_contrastive_kmeans_flag:
            attn_vecs, attn_scores = self.attention(features)  # (N, T, E), (N, T, H, W)
            return {'feature': attn_vecs, 'attn_scores': attn_scores, 'loss_weight': self.loss_weight, 'name': 'vision',
                    'backbone_feature': features, 'mask': learned_mask}

        elif self.config.model_contrastive_supervised_flag:
            attn_vecs, attn_scores = self.attention(features)  # (N, T, E), (N, T, H, W)
            logits = self.cls(attn_vecs)  # (N, T, C)
            pt_lengths = self._get_length(logits)
            return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
                    'attn_scores': attn_scores, 'loss_weight': self.loss_weight, 'name': 'vision',
                    'backbone_feature': features}

        else:
            return {'name': 'vision', 'backbone_feature': features, 'mask': learned_mask}
