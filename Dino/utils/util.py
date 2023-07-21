import numpy as np
import PIL
import cv2

class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def adjust_learning_rate(optimizer, iteration, opt):
    """Decay the learning rate based on schedule"""
    lr = opt.lr
    # stepwise lr schedule
    for milestone in opt.schedule:
        lr *= (
            opt.lr_drop_rate if iteration >= (float(milestone) * opt.num_iter) else 1.0
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = PIL.Image.fromarray(image_numpy)
    image_pil.save(image_path)

def Heatmap(attention_weight,image,map_size):
    (W, H) = map_size
    overlaps = []
    T = attention_weight.shape[0]
    attention_weight = attention_weight.detach().cpu().numpy()
    x = tensor2im(image)
    for t in range(T):
        att_map = attention_weight[t,:,:] # [feature_H, feature_W, 1]
        att_map = cv2.resize(att_map, (W,H)) # [H, W]
        att_map = (att_map*255).astype(np.uint8)
        heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_JET) # [H, W, C]
        overlap = cv2.addWeighted(heatmap, 0.6, x, 0.4, 0)
        overlaps.append(overlap)
    return overlaps