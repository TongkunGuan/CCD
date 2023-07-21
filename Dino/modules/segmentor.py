import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_MLA(nn.Module):
    def __init__(self, in_channels=1024, mla_channels=256):
        super(Conv_MLA, self).__init__()
        self.mla_p2_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p3_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p4_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())

        self.mla_p2 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                              bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                              bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p4 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                              bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())

    def forward(self, res2, res3, res4):
        mla_p4_1x1 = self.mla_p4_1x1(res4)
        mla_p3_1x1 = self.mla_p3_1x1(res3)
        mla_p2_1x1 = self.mla_p2_1x1(res2)

        mla_p3_plus = mla_p4_1x1 + mla_p3_1x1
        mla_p2_plus = mla_p3_plus + mla_p2_1x1

        mla_p4 = self.mla_p4(mla_p4_1x1)
        mla_p3 = self.mla_p3(mla_p3_plus)
        mla_p2 = self.mla_p2(mla_p2_plus)

        return mla_p2, mla_p3, mla_p4

class MLAHead(nn.Module):
    def __init__(self, in_channels=384, mla_channels=128, mlahead_channels=64):
        super(MLAHead, self).__init__()

        self.head2 = nn.Sequential(
            nn.Conv2d(in_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels),
            nn.ReLU(),
            nn.Conv2d(mla_channels, mlahead_channels, 1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU()
          )
        self.head3 = nn.Sequential(
            nn.Conv2d(in_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels),
            nn.ReLU(),
            nn.Conv2d(mla_channels, mlahead_channels, 1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU()
            )
        self.head4 = nn.Sequential(
            nn.Conv2d(in_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels),
            nn.ReLU(),
            nn.Conv2d(mla_channels, mlahead_channels, 1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU()
            )

    def forward(self, mla_p2, mla_p3, mla_p4):
        head2 = self.head2(mla_p2)
        head3 = self.head3(mla_p3)
        head4 = self.head4(mla_p4)
        return torch.cat([head2, head3, head4], dim=1)


class SegHead(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, in_channels=384, mla_channels=128, mlahead_channels=64, num_classes=2, **kwargs):
        super(SegHead, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.conv_mla = Conv_MLA(in_channels, mla_channels)
        self.mlahead = MLAHead(in_channels=in_channels, mla_channels=mla_channels, mlahead_channels=mlahead_channels)
        self.unpool1 = nn.Sequential(nn.ConvTranspose2d(192, 128, (4, 4), (2, 2), (1, 1)),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True))
        self.unpool2 = nn.Sequential(nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1)),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True))
        self.cls = nn.Conv2d(128, self.num_classes, 3, padding=1)

    def forward(self, inputs):
        x = self.mlahead(inputs[0], inputs[1], inputs[2])
        x = self.unpool1(x)
        x = self.unpool2(x)
        x = self.cls(x)
        return x

