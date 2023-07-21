import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Dino.utils.kmeans import run_kmeans

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class ResNetFPN(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 32
        super(ResNetFPN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(32, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.unpool = nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1))
        self.toplayer4 = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        if x.shape == y.shape:
            return x + y
        else:
            _, _, H, W = y.size()
            return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        c0 = self.relu(x)  #torch.Size([64, 32, 32, 128])
        c1 = self.layer1(c0)  #torch.Size([64, 32, 16, 64])
        c2 = self.layer2(c1)  #torch.Size([64, 64, 16, 64])
        c3 = self.layer3(c2)  #torch.Size([64, 128, 8, 32])
        c4 = self.layer4(c3)  #torch.Size([64, 256, 8, 32])
        x = self.layer5(c4)  #torch.Size([64, 512, 8, 32])

        # Top-down
        p4 = self.latlayer1(c4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.toplayer1(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.toplayer2(p2)
        p1 = self._upsample_add(p2, self.latlayer4(c1))
        p1 = self.toplayer3(p1)
        learned_mask = self.toplayer4(self.unpool(p1))
        return x, learned_mask

class ResNetUnet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 32
        super(ResNetUnet, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=1)

        self.unpool1 = nn.Sequential(nn.ConvTranspose2d(256, 256, (4, 4), (2, 2), (1, 1)),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(True))
        self.unpool2 = nn.Sequential(nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1)),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True))
        self.conv0 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(256)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(256, 256, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(384, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 256, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.seg_cls_0 = nn.Sequential(
            nn.Conv2d(256, 2, 3, padding=1),
        )

        self.conv4 = nn.Conv2d(128, 128, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.seg_cls_1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, flag=False):
        x = self.conv(x)
        x = self.bn(x)
        c0 = self.relu(x)  #torch.Size([64, 32, 32, 128])
        c1 = self.layer1(c0)  #torch.Size([64, 32, 16, 64])
        c2 = self.layer2(c1)  #torch.Size([64, 64, 16, 64])

        c3 = self.layer3(c2)  #torch.Size([64, 128, 8, 32])
        c4 = self.layer4(c3)  #torch.Size([64, 256, 8, 32])
        x = self.layer5(c4)  #torch.Size([64, 512, 8, 32])
        if flag:
            h = self.conv0(x)
            h = self.bn0(h)
            h = self.relu0(h)
            h = self.conv1(h)
            h = self.bn1(h)
            h = self.relu1(h)
            # mask_0 = self.seg_cls_0(h)

            g = (self.unpool1(h))  # bs 256 16,64
            c = self.conv2(torch.cat((g, c3), 1))
            c = self.bn2(c)
            c = self.relu2(c)
            h = self.conv3(c)
            h = self.bn3(h)
            h = self.relu3(h)
            mask_1 = self.seg_cls_0(h)

            g = (self.unpool2(h))  # bs 256 32,128
            c = self.conv4(g)
            c = self.bn4(c)
            c = self.relu4(c)

            # nb, nc, nh, nw = c2.shape
            # I = run_kmeans(c2.data.permute(0, 2, 3, 1).reshape(-1, nc).cpu().numpy(), 128, 2, use_pca=False)
            # I = I.reshape(nb, nh, nw)

            mask_2 = self.seg_cls_1(c)

            return x, (mask_1, mask_2)
        else:
            return x, None

def resnet45():
    return ResNet(BasicBlock, [3, 4, 6, 6, 3])

def resnetfpn():
    return ResNetFPN(BasicBlock, [3, 4, 6, 6, 3])

def resnetunet():
    return ResNetUnet(BasicBlock, [3, 4, 6, 6, 3])