from sklearn.cluster import DBSCAN
import numpy as np
import torch.nn as nn
import time
import cv2
from skimage import measure
from scipy import ndimage as ndi


class DBSCAN_cluster(nn.Module):
    def __init__(self, eps=1.5, min_samples=4):
        super(DBSCAN_cluster, self).__init__()

    def forward(self, mask):
        try:
            if mask.sum() != 0:
                ind = np.where(mask > 0.1)
                coordinates = np.hstack([ind[0].reshape(-1, 1), ind[1].reshape(-1, 1)])
                dbscan = DBSCAN(eps=1.5, min_samples=4)
                # dbscan = OPTICS(min_samples=4)
                clusters = dbscan.fit_predict(coordinates)
                # try:
                #     # print((clusters.shape[0], coordinates.shape[0]))
                #     assert clusters.shape[0] == coordinates.shape[0]
                # except:
                #     print('bug')
                # print(f'cluster:{time.time() - t1}')
                zero = np.zeros((26, mask.shape[0], mask.shape[1])).astype(np.uint8)
                # t1 = time.time()
                categorys = np.unique(clusters)
                index = []
                new_category = []
                for category in categorys:
                    if category < 0:
                        continue
                    loc = clusters == category
                    if loc.sum() < 30:
                        clusters[loc] = -3
                    else:
                        index.append(coordinates[loc][:, 1].mean())
                        new_category.append(category)

                index = np.argsort(index)[:26]
                new_index = np.arange(len(index))
                replace_dict = dict(zip(np.array(new_category)[index], new_index))
                # new_cluster = clusters.copy()
                for item in replace_dict.keys():
                    new_item = replace_dict[item]
                    loc = clusters == item
                    # new_cluster[loc] = new_item
                    zero[new_item, coordinates[loc][:, 0], coordinates[loc][:, 1]] = 1
                # print(f'unique:{time.time() - t1}')
            else:
                zero = np.zeros((26, mask.shape[0], mask.shape[1])).astype(np.uint8)
        except:
            print('real error')
            zero = np.zeros((26, mask.shape[0], mask.shape[1])).astype(np.uint8)
        # print(time.time())
        return zero

class label_cluster(nn.Module):
    def __init__(self):
        super(label_cluster, self).__init__()

    def forward(self, mask):
        try:
            if mask.sum() != 0:
                # zero = np.zeros((26, mask.shape[0], mask.shape[1])).astype(np.uint8)
                # cluster = measure.label(mask)
                # regions = measure.regionprops(cluster)
                # for region in regions:
                #     if region.area < 30:
                #         regions.remove(region)
                # regions = sorted(regions, key=lambda x: x.centroid[1], reverse=False)[:26]
                # for index, region in enumerate(regions):
                #     zero[index, cluster == region.label] = 1

                zero_ = np.zeros((26, mask.shape[0], mask.shape[1])).astype(np.uint8)
                zero = np.zeros((26, mask.shape[0], mask.shape[1])).astype(np.uint8)
                cluster = measure.label(mask)
                categorys = np.unique(cluster)
                loc = []
                i = 0
                for cate in categorys:
                    if cate == 0:
                        continue
                    sub_cluster = cluster == cate
                    if sub_cluster.sum() >= 30:
                        loc.append(np.where(sub_cluster)[1].mean())
                        zero_[i, sub_cluster] = 1
                        i += 1
                        if i >= 26:
                            break
                index = np.argsort(loc)
                for i, new_index in enumerate(index):
                    zero[i] = zero_[new_index]
                # zero = np.take(zero_, index, axis=0)
            else:
                zero = np.zeros((26, mask.shape[0], mask.shape[1])).astype(np.uint8)
        except:
            print('real error')
            zero = np.zeros((26, mask.shape[0], mask.shape[1])).astype(np.uint8)
        return zero


class region_cluster(nn.Module):
    def __init__(self):
        super(region_cluster, self).__init__()

    def forward(self, mask):
        try:
            if mask.sum() != 0:
                zero = np.zeros((26, mask.shape[0], mask.shape[1])).astype(np.uint8)
                cluster = measure.label(mask)
                objects = ndi.find_objects(cluster)
                regions = []
                ndim = cluster.ndim
                for slice in objects:
                    bbox = tuple([slice[i].start for i in range(ndim)] +
                                 [slice[i].stop for i in range(ndim)])
                    _ymin, _xmin, _ymax, _xmax = bbox
                    attribute = {
                        'bbox': bbox,
                        'centroid': (_xmax + _xmin) / 2.,
                        'area': (_xmax - _xmin) * (_ymax - _ymin),
                    }
                    regions.append(attribute)
                regions = sorted(regions, key=lambda x: x['centroid'], reverse=False)[:26]
                num = 0
                for region in regions:
                    if region['area'] < 100:
                        continue  # regions.remove(region)
                    _ymin, _xmin, _ymax, _xmax = region['bbox']
                    zero[num, _ymin:_ymax, _xmin:_xmax] = 1
                    num += 1
            else:
                zero = np.zeros((26, mask.shape[0], mask.shape[1])).astype(np.uint8)
        except:
            print('real error')
            zero = np.zeros((26, mask.shape[0], mask.shape[1])).astype(np.uint8)
        return zero
