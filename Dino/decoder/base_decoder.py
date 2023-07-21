# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

class BaseDecoder(nn.Module):
    """Base decoder class for text recognition."""

    def __init__(self):
        super(BaseDecoder, self).__init__()

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def forward_test_speed(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def forward(self,
                feat,
                out_enc,
                targets_dict=None,
                img_metas=None,
                train_mode=True,
                test_speed=False):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets_dict, img_metas)
        if test_speed:
            return self.forward_test_speed(feat, out_enc, img_metas)
        return self.forward_test(feat, out_enc, img_metas)
