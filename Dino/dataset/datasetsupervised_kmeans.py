import random

import imgaug.random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from Dino.utils.kmeans import clusterpixels
from Dino.utils.transforms import ImageToPIL, ImageToArray
from Dino.dataset.dataset import ImageDataset
from Dino.dataset.augmentation_pipelines import get_augmentation_pipeline
from Dino.dataset.transforms import CVGeometry
import cv2
import PIL
import torchvision.transforms.functional as TF
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug
from imgaug.augmenters.geometric import _warp_affine_arr as Affine
class mat:
    params = 0

class ImageDatasetSelfSupervisedKmeans(ImageDataset):
    """
    Image Dataset for Self Supervised training that outputs pairs of images
    """

    def __init__(self, augmentation_severity: int = 1, supervised_flag=False, **kwargs):
        super().__init__(**kwargs)
        self.supervised_flag = supervised_flag
        # self.dbscan = DBSCAN_cluster(eps=1.5, min_samples=4)
        if self.data_aug:
            if augmentation_severity == 0 or (not self.is_training and supervised_flag):
                regular_aug = self.augment_tfs.transforms if hasattr(self, 'augment_tfs') else []
                self.augment_tfs = transforms.Compose([ImageToPIL()] + regular_aug + [ImageToArray()])
            else:
                self.augment_tfs = get_augmentation_pipeline(augmentation_severity).augment_image
                self.augment_color = get_augmentation_pipeline(augmentation_severity+1).augment_image
                self.augment_geo = iaa.Affine(
                    scale={"x": (0.6, 1.1), "y": (0.6, 1.1)},
                    translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                    rotate=(-10, 10),
                    shear={"x": (-45, 45), "y": (-10, 10)}
                )
                self.random_state = imgaug.random.RNG(1234)

    def _process_training(self, image, mask):
        # mask = clusterpixels(image.convert("L"), 2)
        image = np.array(image)
        image_views = []
        for _ in range(3):
            if self.data_aug:
                try:
                    if _ == 0:
                        image_view = image
                    else:
                        image_view = self.augment_tfs(image)
                        if _ == 2:
                            if random.random() > 0.3:
                                batch = imgaug.augmentables.batches._BatchInAugmentation(images=[image_view])
                                samples = self.augment_geo._draw_samples(batch.nb_rows, self.random_state)
                                (image_view_, matric) = self.augment_geo._augment_images_by_samples(batch.images, samples, return_matrices=True)
                                image_view = image_view_[0]
                                W_scale = image.shape[1] / self.img_w
                                H_scale = image.shape[0] / self.img_h
                                W_inv = np.array([[1 / W_scale, 0, 0], [0, 1 / H_scale, 0], [0, 0, 1]])
                                W = np.array([[W_scale, 0, 0], [0, H_scale, 0], [0, 0, 1]])
                                metric = np.matmul(np.matmul(W_inv, matric[0]._inv_matrix), W)
                                W_ = np.array([[2 / (self.img_w - 1), 0, -1], [0, 2 / (self.img_h - 1), -1], [0, 0, 1]])
                                theta = np.matmul(np.matmul(W_, metric), np.matrix(W_).I)
                            else:
                                image_view = image
                                theta = np.diag(np.ones(3))
                except:
                    print('unknown error')
                    image_view = image
                    theta = np.diag(np.ones(3))
            else:
                image_view = image
                theta = np.diag(np.ones(3))
            image_view = self.totensor(self.resize(image_view))
            image_view = TF.normalize(image_view, self.mean, self.std)
            image_views.append(image_view)
        mask_view = cv2.resize(mask.astype(np.float32), (self.img_w, self.img_h))
        mask_view = (mask_view >= 0.5).astype(np.float32)
        return np.stack(image_views, axis=0), mask_view, np.array(theta, dtype=np.float32)


    def _process_test(self, image):
        return self._process_training(image)

    def _postprocessing(self, image, mask, metric, idx):
        return image, mask, metric
