import logging
import re

import cv2
import lmdb
import six
from fastai.vision import *
from torchvision import transforms
from torch.utils.data.dataloader import default_collate

from Dino.utils.transforms import CVColorJitter, CVDeterioration, CVGeometry
from Dino.utils.utils import CharsetMapper, onehot
import torchvision.transforms.functional as TF
from Dino.convertor.attn import AttnConvertor
from imgaug import augmenters as iaa


class ImageDataset(Dataset):
    "`ImageDataset` read data from LMDB database."

    def __init__(self,
                 path: PathOrStr,
                 is_training: bool = True,
                 img_h: int = 32,
                 img_w: int = 100,
                 max_length: int = 25,
                 check_length: bool = True,
                 filter_single_punctuation: bool = False,
                 case_sensitive: bool = False,
                 type: str = 'DICT90',
                 convert_mode: str = 'RGB',
                 data_aug: bool = True,
                 deteriorate_ratio: float = 0.,
                 multiscales: bool = True,
                 one_hot_y: bool = True,
                 return_idx: bool = False,
                 return_raw: bool = False,
                 data_portion: float = 1.0,
                 mask: bool = False,
                 use_abi: bool = False,
                 **kwargs):
        self.path, self.name = Path(path), Path(path).name
        assert self.path.is_dir() and self.path.exists(), f"{path} is not a valid directory."
        self.convert_mode, self.check_length = convert_mode, check_length
        self.img_h, self.img_w = img_h, img_w
        self.max_length, self.one_hot_y = max_length, one_hot_y
        self.return_idx, self.return_raw = return_idx, return_raw
        self.case_sensitive, self.is_training = case_sensitive, is_training
        self.filter_single_punctuation = filter_single_punctuation
        self.data_aug, self.multiscales, self.mask = data_aug, multiscales, mask
        self.label_convertor = AttnConvertor(dict_type=type, max_seq_len=max_length, with_unknown=True)
        self.use_abi = use_abi

        self.env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        assert self.env, f'Cannot open LMDB dataset from {path}.'
        with self.env.begin(write=False) as txn:
            dataset_length = int(txn.get('num-samples'.encode()))

        print(f'current_dataset_path:{str(path)}-->{dataset_length}')
        logging.info(repr(f'current_dataset_path:{str(path)}-->{dataset_length}') + "\n")
        self.use_portion = self.is_training and not data_portion == 1.0
        if not self.use_portion:
            self.length = dataset_length
        else:
            self.length = int(data_portion * dataset_length)
            self.optional_ind = np.random.permutation(dataset_length)[:self.length]

        if self.is_training and self.data_aug:
            if self.use_abi:
                '''source augmentation ABINet'''
                self.augment_tfs = transforms.Compose([
                    CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                    CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                    CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
                ])
                print('source augmentation ABINet')
            else:
                # for finetune ViT-Small select ['0.6', '0.8', '0.6', '0.6', '0.6']
                # for finetune ViT-Base select ['0.4', '0.7', '0.7', '0.7', '0.5']
                augmentations = iaa.Sequential([
                    iaa.Sometimes(0.6,
                                  iaa.Invert(0.1),
                                  ),
                    iaa.Sometimes(0.8,
                                  iaa.OneOf([
                                    iaa.ChannelShuffle(0.35),
                                    iaa.AddElementwise((-40, 40)),  # 0.020992517471313477
                                    iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),  # 0.03757429122924805
                                    iaa.AdditiveLaplaceNoise(scale=(0, 0.2 * 255)),  # 0.028045654296875
                                    iaa.AdditivePoissonNoise(lam=(0, 40)),  # 0.02863311767578125
                                    iaa.ImpulseNoise(0.1),  # 0.019927501678466797
                                    iaa.Multiply((0.5, 1.5), per_channel=0.5),  # 0.011006593704223633
                                    iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5),  # 0.022083520889282227
                                    iaa.Dropout(p=(0, 0.1), per_channel=0.5),  # 0.013317108154296875
                                    iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),  # 0.02934122085571289
                                    iaa.Dropout2d(p=0.5),  # 0.00327301025390625
                                    iaa.SaltAndPepper(0.1),  # 0.02179861068725586
                                    iaa.Salt(0.1),  # 0.0442655086517334
                                    iaa.Pepper(0.1),  # 0.023215055465698242
                                    iaa.Solarize(0.5, threshold=(32, 128)),  # 0.0045392513275146484
                                    iaa.JpegCompression(compression=(70, 99)),  # 0.06881833076477051
                                    iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),  # 0.01827859878540039
                                    iaa.EdgeDetect(alpha=(0.0, 1.0)),  # 0.01613450050354004
                                    iaa.DirectedEdgeDetect(alpha=(0.0, 1.0), direction=(0.0, 1.0)),  # 0.06784319877624512
                                    iaa.pillike.FilterEdgeEnhanceMore(),  # 0.008636713027954102
                                    iaa.pillike.FilterContour(),  # 0.008586645126342773
                                    iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                                       children=iaa.WithChannels(0, iaa.Add((0, 50)))),  # 0.007066011428833008
                                    iaa.MultiplyBrightness((0.5, 1.5)),
                                    iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),  # 0.010883331298828125
                                    iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),  # 0.00563359260559082
                                    iaa.AddToHueAndSaturation((-50, 50), per_channel=True),  # 0.01298975944519043
                                    iaa.Sequential([
                                        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                                        iaa.WithChannels(0, iaa.Add((50, 100))),
                                        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
                                    ]),  # 0.002664804458618164
                                    iaa.Grayscale(alpha=(0.0, 1.0)),  # 0.0020821094512939453
                                    iaa.KMeansColorQuantization(),  # 0.13840675354003906
                                    iaa.UniformColorQuantization(),  # 0.004789829254150391
                                    iaa.ChangeColorTemperature((1100, 10000)),  # 0.0026831626892089844
                                    iaa.Fog(),  # 0.009767293930053711
                                    iaa.Clouds(),  # 0.020981788635253906
                                    iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),  # 0.024015426635742188
                                    iaa.Rain(speed=(0.1, 0.3)),  # 0.02486562728881836
                                  ])),
                    iaa.Sometimes(0.6,
                                  iaa.OneOf([
                                      iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)),
                                      iaa.OneOf([
                                          iaa.GaussianBlur((0.5, 1.5)),
                                          iaa.AverageBlur(k=(2, 6)),
                                          iaa.MedianBlur(k=(3, 7)),
                                          iaa.MotionBlur(k=5)
                                      ])
                                  ])),
                    iaa.Sometimes(0.6,
                                  iaa.OneOf([
                                    iaa.GammaContrast((0.5, 2.0)),  # 0.0015556812286376953
                                    iaa.LinearContrast((0.5, 1.0)),  # 0.001466512680053711
                                    iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),  # 0.001722097396850586
                                    iaa.LogContrast(gain=(0.6, 1.4)),  # 0.0016601085662841797
                                    iaa.HistogramEqualization(),  # 0.001706838607788086
                                    iaa.AllChannelsHistogramEqualization(),  # 0.0014772415161132812
                                    iaa.CLAHE(),  # 0.009737253189086914
                                    iaa.AllChannelsCLAHE(),  # 0.012245655059814453
                                  ])),
                    iaa.Sometimes(0.6,
                                  iaa.OneOf([
                                      iaa.Affine(
                                          scale={"x": (0.6, 1.1), "y": (0.6, 1.1)},
                                          translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                                          rotate=(-10, 10),
                                          shear={"x": (-45, 45), "y": (-10, 10)}
                                      ),
                                      iaa.PiecewiseAffine(scale=(0.01, 0.1)),
                                      iaa.Rotate((-45, 45))
                                  ])),
                ])
                self.augment_tfs = augmentations.augment_image
        self.totensor = transforms.ToTensor()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return self.length

    def _next_image(self):
        if not self.is_training:
            return
        next_index = random.randint(0, len(self) - 1)
        if self.use_portion:
            next_index = self.optional_ind[next_index]
        return self.get(next_index)

    def _check_image(self, x, pixels=6):
        if x.size[0] <= pixels or x.size[1] <= pixels:
            return False
        else:
            return True

    def resize_multiscales(self, img, borderType=cv2.BORDER_CONSTANT):
        def _resize_ratio(img, ratio, fix_h=True):
            if ratio * self.img_w < self.img_h:
                if fix_h:
                    trg_h = self.img_h
                else:
                    trg_h = int(ratio * self.img_w)
                trg_w = self.img_w
            else:
                trg_h, trg_w = self.img_h, int(self.img_h / ratio)
            img = cv2.resize(img, (trg_w, trg_h))
            pad_h, pad_w = (self.img_h - trg_h) / 2, (self.img_w - trg_w) / 2
            top, bottom = math.ceil(pad_h), math.floor(pad_h)
            left, right = math.ceil(pad_w), math.floor(pad_w)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType)
            return img

        if self.is_training:
            if random.random() < 0.5:
                base, maxh, maxw = self.img_h, self.img_h, self.img_w
                h, w = random.randint(base, maxh), random.randint(base, maxw)
                return _resize_ratio(img, h / w)
            else:
                return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio
        else:
            return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio

    def resize(self, img):
        if self.multiscales:
            return self.resize_multiscales(img, cv2.BORDER_REPLICATE)
        else:
            return cv2.resize(img, (self.img_w, self.img_h))

    def get(self, idx):
        with self.env.begin(write=False) as txn:
            image_key, label_key = f'image-{idx + 1:09d}', f'label-{idx + 1:09d}'
            try:
                raw_label = str(txn.get(label_key.encode()), 'utf-8')  # label
                if self.is_training and len(raw_label) == 0:
                    return self._next_image()
                if self.is_training:
                    targets_dict = self.label_convertor.str2tensor([raw_label])
                    ### process ARD/Openimages exist '' label.
                    if targets_dict[0][0] == targets_dict[0][1] and targets_dict[0][1] == 91:
                        print(raw_label)
                        return self._next_image()
                else:
                    targets_dict = [raw_label]
                imgbuf = txn.get(image_key.encode())  # image
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
                    image = PIL.Image.open(buf).convert(self.convert_mode)
                if self.is_training and not self._check_image(image):
                    # print(image.size)
                    # logging.info(f'Invalid image is found: {self.name}, {idx}')
                    return self._next_image()
            except:
                # import traceback
                # traceback.print_exc()
                # if "label" in locals():
                #     logging.info(f'Corrupted image is found: {self.name}, {idx}, {label}, {len(label)}')
                # else:
                #     logging.info(f'Corrupted image is found: {self.name}, {idx}')
                return self._next_image()
            return image, targets_dict

    def _process_training(self, image):
        if self.data_aug: image = self.augment_tfs(image)
        image = self.totensor(self.resize(np.array(image)))
        image = TF.normalize(image, self.mean, self.std)
        return image

    def _process_test(self, image):
        image = self.totensor(self.resize(np.array(image)))  # TODO:move is_training to here
        image = TF.normalize(image, self.mean, self.std)
        return image

    def __getitem__(self, idx):
        if self.use_portion:
            idx = self.optional_ind[idx]
        datum = self.get(idx)
        if datum is None:
            return

        image, text = datum
        if not self.use_abi:
            image = np.array(image)
        if self.is_training:
            image = self._process_training(image)
        else:
            image = self._process_test(image)
        return self._postprocessing(image, text)

    def _postprocessing(self, image, text):
        return image, text


def collate_fn_filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)
