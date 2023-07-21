import torch
import torch.nn as nn
import torch.nn.functional as F
import logging, time
import numpy as np
from fastai.vision import *

from Dino.modules import vision_transformer as vits
from torchvision import models as torchvision_models
from Dino.decoder.nrtr_decoder import NRTRDecoder
from Dino.convertor.attn import AttnConvertor
from Dino.loss.ce_loss import TFLoss

from Dino.modules.model_vision import BaseVision
from Dino.modules.model import Model, _default_tfmer_cfg
from Dino.modules.attention import *
from Dino.utils.DBSCAN import DBSCAN_cluster, label_cluster
dbscan = DBSCAN_cluster(eps=1.5, min_samples=4)
label = label_cluster()

class ABIDINOModel(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, Segmentation, head):
        super(ABIDINOModel, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.segmentation = Segmentation
        self.head = head

    def attention(self, feature, clusters):
        N, E, H, W = feature.size()
        clusters = nn.functional.interpolate(clusters, size=(H, W), scale_factor=None, mode='bilinear',
                                             align_corners=None)
        max_cluster_index = clusters.reshape(N, 26, -1).sum(-1)
        new_cluster = clusters / max_cluster_index.unsqueeze(-1).unsqueeze(-1)
        new_cluster[torch.isnan(new_cluster)] = 0

        v = feature.permute(0, 2, 3, 1).view(N, -1, E)  # (N, (H*W), E)
        attn_vecs = torch.bmm(new_cluster.view(N, 26, -1), v)  # (N, T, E)
        index = max_cluster_index > 0
        return attn_vecs, index

    def forward(self, x, metrics, target_mask, epoch, clusters=None, index=None):
        x1 = x[:, 0]
        x2 = x[:, 1]
        backbone_out, Segmentation_input = self.backbone(torch.cat([x1, x2]))
        N, T, E = backbone_out.shape
        region_f = backbone_out.reshape(N, 8, 32, E).permute(0, 3, 1, 2)
        if clusters is None:
            Segmentation_out = self.segmentation(Segmentation_input)
            if epoch < 30:
                B, H, W = target_mask.shape
                zero = np.zeros((B, 26, H, W))
                for i, mask in enumerate(target_mask.cpu().numpy()):
                    zero[i] = label(mask)
            else:
                backfore_softmax0 = F.softmax(Segmentation_out, dim=1)
                target_mask0 = (backfore_softmax0[:, 1, :, :] > 0.5).int().cpu().detach().numpy()
                B, H, W = target_mask0.shape
                zero = np.zeros((B // 2, 26, H, W))
                for i, mask in enumerate(target_mask0[:B // 2]):
                    zero[i] = label(mask)
            clusters_source = torch.Tensor(zero).to(Segmentation_out.device)
            affine_grid = F.affine_grid(metrics[:, :2, :],
                                        size=(
                                            clusters_source.shape[0], 1, clusters_source.shape[2],
                                            clusters_source.shape[3]))
            clusters_image = F.grid_sample(clusters_source, affine_grid.to(clusters_source.device))
            clusters_image = (clusters_image > 0.1).float()
            clusters = torch.cat([clusters_source, clusters_image], dim=0)

            region_attention_out, index = self.attention(region_f, clusters)

            B = index.shape[0]
            length = torch.clamp(index.sum(1), 3, 26).unsqueeze(-1)[:B // 2]
            grid = torch.arange(0, 26, device=length.device).unsqueeze(0)
            new_index = grid <= length

            region_attention_out = torch.cat([region_attention_out[: B // 2][new_index], region_attention_out[B // 2:][new_index]])
            region_out = self.head(region_attention_out)

            # from Dino.utils.kmeans import run_kmeans
            # I = run_kmeans(attention_out.reshape(-1,192).detach().cpu().numpy(), pca=128, nmb_clusters=12, use_pca=False)
            res = {'instances_view': region_out,
                   'mask': Segmentation_out,
                   'image': x,
                   'zero': clusters,
                   'index': new_index
                   }
        else:
            region_attention_out, index = self.attention(region_f, clusters)

            B = index.shape[0]
            length = torch.clamp(index.sum(1), 3, 26).unsqueeze(-1)[:B // 2]
            grid = torch.arange(0, 26, device=length.device).unsqueeze(0)
            new_index = grid <= length

            region_attention_out = torch.cat(
                [region_attention_out[: B // 2][new_index], region_attention_out[B // 2:][new_index]])

            region_out = self.head(region_attention_out)

            res = {'instances_view': region_out,
                   'feature': region_f,
                   }

        return res

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
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

class DINO_Finetune(nn.Module):
    def __init__(self, config):
        super(DINO_Finetune, self).__init__()
        self.label_convertor = AttnConvertor(dict_type='DICT90', max_seq_len=config.decoder_max_seq_len, with_unknown=True)

        """ model configuration """
        # ============ building student and teacher networks ... ============
        # we changed the name DeiT-S for ViT-S to avoid confusions
        config.arch = config.arch.replace("deit", "vit")
        # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
        if config.arch in vits.__dict__.keys():
            self.backbone = vits.__dict__[config.arch](
                patch_size=config.patch_size,
                drop_path_rate=config.drop_path_rate,  # stochastic depth
            )
            embed_dim = self.backbone.embed_dim
        # if the network is a XCiT
        elif config.arch in torch.hub.list("facebookresearch/xcit:main"):
            self.backbone = torch.hub.load('facebookresearch/xcit:main', config.arch,
                                     pretrained=False, drop_path_rate=config.drop_path_rate)
            embed_dim = self.backbone.embed_dim
        # otherwise, we check if the architecture is in torchvision models
        elif config.arch in torchvision_models.__dict__.keys():
            self.backbone = torchvision_models.__dict__[config.arch]()
            embed_dim = self.backbone.fc.weight.shape[1]
        else:
            print(f"Unknow architecture: {config.arch}")

        self.encoder = Mlp(in_features=embed_dim, hidden_features=512, out_features=512, act_layer=nn.GELU, drop=0.1)
        # self.encoder = None
        config.decoder_num_classes=self.label_convertor.num_classes()
        config.decoder_start_idx=self.label_convertor.start_idx
        config.decoder_padding_idx=self.label_convertor.padding_idx
        self.decoder = NRTRDecoder(
            n_layers=config.decoder_n_layers,
            d_embedding=config.decoder_d_embedding,
            n_head=config.decoder_n_head,
            d_k=config.decoder_d_k,
            d_v=config.decoder_d_v,
            d_model=config.decoder_d_model,
            d_inner=config.decoder_d_inner,
            n_position=200,
            dropout=0.1,
            num_classes=config.decoder_num_classes,
            max_seq_len=config.decoder_max_seq_len,
            start_idx=config.decoder_start_idx,
            padding_idx=config.decoder_padding_idx, )

        self.loss = TFLoss(ignore_index=self.label_convertor.padding_idx)

    def forward(self, img, text, return_loss=True, test_speed=False):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.
        """

        if return_loss:
            return self.forward_train(img, text)

        if test_speed:
            return self.forward_test_speed(img)

        return self.forward_test(img)

    def extract_feat(self, img):
        """Directly extract features from the backbone."""
        x, out = self.backbone(img)
        # x = x[:, 1:]
        return x

    def forward_train(self, img, img_metas):
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        feat = self.extract_feat(img)

        # gt_labels = img_metas['text']
        # targets_dict = self.label_convertor.str2tensor(gt_labels)
        targets_dict = {'padded_targets': img_metas}

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)
        else:
            out_enc = feat

        out_dec, attn = self.decoder(feat, out_enc, targets_dict, train_mode=True)

        loss_inputs = (out_dec, targets_dict,)
        losses = self.loss(*loss_inputs)

        return losses, attn

    def forward_test(self, img):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        feat = self.extract_feat(img)

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)
        else:
            out_enc = feat

        out_dec = self.decoder(feat, out_enc, None, train_mode=False)

        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return out_dec

        # flatten batch results
        # results = []
        # for string, score in zip(label_strings, label_scores):
        #     results.append(dict(text=string, score=score))

        return out_dec

    def forward_test_speed(self, img):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        feat = self.extract_feat(img)

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)
        else:
            out_enc = feat

        out_dec = self.decoder(feat, out_enc, None, train_mode=False, test_speed=True)

        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return out_dec

        # flatten batch results
        # results = []
        # for string, score in zip(label_strings, label_scores):
        #     results.append(dict(text=string, score=score))

        return out_dec