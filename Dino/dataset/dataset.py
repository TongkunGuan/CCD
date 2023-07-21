import logging
import re
import time

import cv2
import lmdb
import six
from fastai.vision import *
from torchvision import transforms
from torch.utils.data.dataloader import default_collate

from Dino.utils.transforms import CVColorJitter, CVDeterioration, CVGeometry
from Dino.utils.utils import CharsetMapper, onehot
from Dino.utils.kmeans import clusterpixels

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
                 charset_path: str = 'data/charset_36.txt',
                 convert_mode: str = 'RGB',
                 data_aug: bool = True,
                 deteriorate_ratio: float = 0.,
                 multiscales: bool = True,
                 one_hot_y: bool = True,
                 return_idx: bool = False,
                 return_raw: bool = False,
                 data_portion: float = 1.0,
                 mask: bool = False,
                 mask_path: str = '',
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
        self.charset = CharsetMapper(charset_path, max_length=max_length + 1)
        self.charset_string = ''.join([*self.charset.char_to_label])
        self.charset_string = re.sub('-', r'\-', self.charset_string)  # escaping the hyphen for later use in regex
        self.c = self.charset.num_classes

        self.env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        assert self.env, f'Cannot open LMDB dataset from {path}.'
        try:
            sub_file = str(path).split('training')[1]
            self.mask_env = lmdb.open(mask_path+sub_file,
                                      readonly=True, lock=False, readahead=False, meminit=False)
            assert self.mask_env, f'Cannot open LMDB dataset from {path}.'
        except:
            print(f'{str(path)} not use loading mask lmdb file!')
        with self.env.begin(write=False) as txn:
            dataset_length = int(txn.get('num-samples'.encode()))
        self.use_portion = self.is_training and not data_portion == 1.0
        if not self.use_portion:
            self.length = dataset_length
        else:
            self.length = int(data_portion * dataset_length)
            self.optional_ind = np.random.permutation(dataset_length)[:self.length]

        if self.is_training and self.data_aug:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])
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
                imgbuf = txn.get(image_key.encode())  # image
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
                    image = PIL.Image.open(buf).convert(self.convert_mode)
                with self.mask_env.begin(write=False) as mask_txn:
                    mask_key = f'mask-{idx + 1:09d}'
                    try:
                        maskbuf = mask_txn.get(mask_key.encode())  # image
                        mask_buf = six.BytesIO()
                        mask_buf.write(maskbuf)
                        mask_buf.seek(0)
                        mask = PIL.Image.open(mask_buf).convert('L')
                    except:
                        print(f"Corrupted image for {idx}")
                        mask = np.zeros((self.img_w, self.img_h))
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
        return image, mask, idx

    # nparr = np.fromstring(buf, np.uint8)
    # img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    def _process_training(self, image):
        if self.data_aug: image = self.augment_tfs(image)
        image = self.totensor(self.resize(np.array(image)))
        return image

    def _process_test(self, image):
        return self.totensor(self.resize(np.array(image)))  # TODO:move is_training to here

    def __getitem__(self, idx):
        if self.use_portion:
            idx = self.optional_ind[idx]
        datum = self.get(idx)
        if datum is None:
            return
        image, mask, idx_new = datum

        if self.mask:
            if self.is_training:
                image, mask, metric = self._process_training(image, mask)
            else:
                image, mask, metric = self._process_test(image)
            return self._postprocessing(image, mask, metric, idx_new)
        else:
            if self.is_training:
                image = self._process_training(image)
            else:
                image = self._process_test(image)
            return self._postprocessing(image, text, idx_new)

    def _postprocessing(self, image, text, idx):
        if self.return_raw: return image, text

        length = tensor(len(text) + 1).to(dtype=torch.long)  # one for end token
        label = self.charset.get_labels(text, case_sensitive=self.case_sensitive)
        label = tensor(label).to(dtype=torch.long)
        if self.one_hot_y: label = onehot(label, self.charset.num_classes)

        if self.return_idx:
            y = [label, length, idx]
        else:
            y = [label, length]
        return image, y


def collate_fn_filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)
