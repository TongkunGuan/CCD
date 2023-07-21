import math
import torch
import matplotlib.pyplot as plt
irange = range
from scipy.cluster.vq import *
from pylab import *
from PIL import Image
import numpy as np
import cv2
def make_grid(tensor, normalize=True, scale_each=True, range=None, nrow=4, padding=2, pad_value=0):
    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid[0, y * height + padding: (y + 1) * height, x * width + padding: (x + 1) * width] = tensor[
                y * xmaps + x]
            # grid.narrow(1, y * height + padding, height - padding) \
            #     .narrow(2, x * width + padding, width - padding) \
            #     .copy_(tensor[k])
            k = k + 1
    return grid


def remake_attn(Attention,map_size,cls_threshold):
    h, w = map_size
    H, W = Attention.shape
    Attention = Attention.detach().cpu()
    char_index = torch.argmax(Attention,1)
    output = torch.zeros(h,w)
    for i in range(W):
        try:
            index = torch.where(char_index == i)[0]
            secone_index = index[Attention[index].max(1)[0]>cls_threshold]
            output[secone_index//w,secone_index%w] = Attention[secone_index].max(1)[0]
        except:
            print("")
    return output
def get_boundry_index(map_size):
    k,h,w = map_size
    index = [(0,i) for i in range(w)]
    index += [(h - 1,i) for i in range(w)]
    index += [(i, 0) for i in range(1,h-1)]
    index += [(i, w-1) for i in range(1,h-1)]
    x = []
    y = []
    for coordinate in index:
        x.append(coordinate[0])
        y.append(coordinate[1])
    return (x,y)
def remake_multi_attn(Attention,map_size,cls_threshold):
    k, h, w = map_size
    H, W = Attention.shape
    Attention = Attention.detach().cpu()
    output = torch.zeros(k,h,w)
    boundry = get_boundry_index(map_size)
    output[:, boundry[0], boundry[1]] = 1
    for i in range(W):
        try:
            index = torch.where(Attention[:,i] > cls_threshold)[0]
            output[i,index//w,index%w] = Attention[:,i][index]
        except:
            print("")
    return output

def clusterpixels(im, k):
    im = np.array(im)
    h, w = im.shape
    im = im.astype(np.float).reshape(-1)
    # 聚类， k是聚类数目
    centroids, variance = kmeans(im, k)
    code, distance = vq(im, centroids)
    code = code.reshape(h, w)
    fc = sum(code[:, 0])
    lc = sum(code[:, -1])
    fr = sum(code[0, :])
    lr = sum(code[-1, :])
    num = int(fr > w // 2) + int(lr > w // 2) + int(fc > h // 2) + int(lc > h // 2)
    if num >= 3:
        return 1 - code
    else:
        return code
def clusterpixels1(im, k):
    h, w = im.shape
    im = im.astype(np.float).reshape(-1)
    # 聚类， k是聚类数目
    centroids, variance = kmeans(im, k)
    code, distance = vq(im, centroids)
    code = code.reshape(h, w)
    fc = sum(code[:, 0])
    lc = sum(code[:, -1])
    fr = sum(code[0, :])
    lr = sum(code[-1, :])
    num = int(fr > w // 2) + int(lr > w // 2) + int(fc > h // 2) + int(lc > h // 2)
    if num >= 3:
        return 1 - code
    else:
        return code

def meshgrid(x, y, row_major=True):
    '''Return meshgrid in range x & y.
    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.
    Returns:
      (tensor) meshgrid, sized [x*y,2]
    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]
    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    '''
    a = torch.arange(0,x)  #v3
    b = torch.arange(0,y)        #v3

    xx = a.repeat(y).view(-1,1)
    yy = b.view(-1,1).repeat(1,x).view(-1,1)
    return torch.cat([xx,yy],1).float() if row_major else torch.cat([yy,xx],1).float()
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)
def Heatmap(attention_weight,image,map_size):
    (W, H) = map_size
    overlaps = []
    T = attention_weight.shape[0]
    attention_weight_ = attention_weight.detach().cpu().numpy()
    x = tensor2im(image)
    for t in range(T):
        att_map = attention_weight_[t,:,:] # [feature_H, feature_W, 1]
        att_map = cv2.resize(att_map, (W,H)) # [H, W]
        att_map = (att_map*255).astype(np.uint8)
        heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_OCEAN) # [H, W, C]
        overlap = cv2.addWeighted(heatmap, 1.0, x, 0.0, 0)
        overlaps.append(overlap)
    return overlaps