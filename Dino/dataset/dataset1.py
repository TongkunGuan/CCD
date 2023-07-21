import os
import sys
import re
import six
import random

from natsort import natsorted
import PIL
import lmdb
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from modules.utils import clusterpixels
from data_process.transform import Compose_, CVGeometry, CVColorJitter, CVDeterioration

class Batch_Balanced_Dataset(object):
    def __init__(
        self, opt, dataset_root, select_data, batch_ratio, log, learn_type=None
    ):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        self.opt = opt
        dashed_line = "-" * 80
        print(dashed_line)
        log.write(dashed_line + "\n")
        print(
            f"dataset_root: {dataset_root}\nselect_data: {select_data}\nbatch_ratio: {batch_ratio}"
        )
        log.write(
            f"dataset_root: {dataset_root}\nselect_data: {select_data}\nbatch_ratio: {batch_ratio}\n"
        )
        assert len(select_data) == len(batch_ratio)


        _AlignCollate = AlignCollate(self.opt)
        data_type = "label"

        self.data_list = []
        self.dataloader_iter_list = []
        for selected_d, batch_ratio_d in zip(select_data, batch_ratio):
            print(dashed_line)
            log.write(dashed_line + "\n")
            _dataset, _dataset_log = hierarchical_dataset(
                root=dataset_root,
                opt=self.opt,
                select_data=[selected_d],
                data_type=data_type,
            )
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            self.data_list.append(_dataset)

        _dataset = ConcatDataset(self.data_list)
        self.dataloader = torch.utils.data.DataLoader(
            _dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=int(self.opt.workers),
            collate_fn=_AlignCollate,
            pin_memory=False,
            drop_last=False,
        )
        self.data_loader_iter = iter(self.dataloader)
    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_labels = []
        balanced_batch_masks = []

        try:
            image, label, mask = self.data_loader_iter.next()
            balanced_batch_images.append(image)
            balanced_batch_masks.append(mask)
            balanced_batch_labels += label
        except StopIteration:
            self.data_loader_iter = iter(self.dataloader)
            image, label, mask = self.data_loader_iter.next()
            balanced_batch_images.append(image)
            balanced_batch_masks.append(mask)
            balanced_batch_labels += label
        except ValueError:
            print('error'*100)
            pass
        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        balanced_batch_masks = torch.cat(balanced_batch_masks, 0)

        return balanced_batch_images, balanced_batch_labels, balanced_batch_masks

def hierarchical_dataset(root, opt, select_data="/", data_type="label", mode="train"):
    """select_data='/' contains all sub-directory of root directory"""
    dataset_list = []
    dataset_log = f"dataset_root:    {root}\t dataset: {select_data[0]}"
    print(dataset_log)
    dataset_log += "\n"
    for dirpath, dirnames, filenames in os.walk(root + "/"):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                if data_type == "label":
                    dataset = LmdbDataset(dirpath, opt, mode=mode)
                else:
                    dataset = LmdbDataset_unlabel(dirpath, opt)
                sub_dataset_log = f"sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}"
                print(sub_dataset_log)
                dataset_log += f"{sub_dataset_log}\n"
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):
    def __init__(self, root, opt, mode="train"):

        self.root = root
        self.opt = opt
        self.mode = mode
        self.env = lmdb.open(
            root,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            print("cannot open lmdb from %s" % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get("num-samples".encode()))
            self.filtered_index_list = []
            for index in range(self.nSamples):
                index += 1  # lmdb starts with 1
                label_key = "label-%09d".encode() % index
                label = txn.get(label_key).decode("utf-8")

                # length filtering
                length_of_label = len(label)
                if length_of_label > opt.batch_max_length:
                    continue

                self.filtered_index_list.append(index)

            self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = "label-%09d".encode() % index
            label = txn.get(label_key).decode("utf-8")
            img_key = "image-%09d".encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                image = PIL.Image.open(buf)
                img = image.convert("RGB")
                mask = clusterpixels(image.convert("L"),2)

            except IOError:
                print(f"Corrupted image for {index}")
                # make dummy image and dummy label for corrupted image.
                img = PIL.Image.new("RGB", (self.opt.imgW, self.opt.imgH))
                label = "[dummy_label]"
                mask = PIL.Image.new("L", (self.opt.imgW, self.opt.imgH))

            if not self.opt.sensitive:
                label = label.lower()
                label = re.sub(f'[^{self.opt.character}]', '', label)

        return (img, mask, label)


class LmdbDataset_unlabel(Dataset):
    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(
            root,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            print("cannot open lmdb from %s" % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get("num-samples".encode()))
            self.index_list = [index + 1 for index in range(self.nSamples)]

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        index = self.index_list[index]

        with self.env.begin(write=False) as txn:
            img_key = "image-%09d".encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = PIL.Image.open(buf).convert("RGB")

            except IOError:
                print(f"Corrupted image for {img_key}")
                # make dummy image for corrupted image.
                img = PIL.Image.new("RGB", (opt.imgW, opt.imgH))

        return img


class RawDataset(Dataset):
    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == ".jpg" or ext == ".jpeg" or ext == ".png":
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            img = PIL.Image.open(self.image_path_list[index]).convert("RGB")

        except IOError:
            print(f"Corrupted image for {index}")
            # make dummy image and dummy label for corrupted image.
            img = PIL.Image.new("RGB", (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])

class ResizeNormalize(object):
    def __init__(self, size, interpolation=PIL.Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        # img.sub_(0.5).div_(0.5)
        img = TF.normalize(img, self.mean, self.std)
        return img

class Resize(object):
    def __init__(self, size, interpolation=PIL.Image.BICUBIC):
        self.size = size
        self.toTensor = transforms.ToTensor()
        self.interpolation = interpolation

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        return img

class AlignCollate(object):
    def __init__(self, opt, mode="train"):
        self.opt = opt
        self.mode = mode

        if self.opt.Aug and self.mode == "train":
            self.augment_tfs = Compose_([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])
        self.transform = ResizeNormalize((opt.imgW, opt.imgH))
        self.mask_transfrom = Resize((opt.imgW, opt.imgH))

    def __call__(self, batch):
        images, masks, labels = zip(*batch)
        image_tensors = []
        mask_tensors = []
        for image, gray in zip(images, masks):
            mask = PIL.Image.fromarray(gray)
            if self.opt.Aug and self.mode == "train":
                image, mask = self.augment_tfs(image, mask)
            image_tensors.append(self.transform(image))
            mask_tensors.append(self.mask_transfrom(mask))
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        mask_tensors = torch.cat([t.float().unsqueeze(0) for t in mask_tensors], 0)
        return image_tensors, labels, mask_tensors


class AlignCollate_SemiSL(object):
    def __init__(self, opt):
        self.opt = opt
        self.transform = Text_augment(opt)

    def __call__(self, batch):

        if "MeanT" in self.opt.semi:
            student_list = []
            teacher_list = []
            for image in batch:
                student_data = self.transform(image)
                teacher_data = self.transform(image)
                student_list.append(student_data)
                teacher_list.append(teacher_data)

            student_tensors = torch.cat([t.unsqueeze(0) for t in student_list], 0)
            teacher_tensors = torch.cat([t.unsqueeze(0) for t in teacher_list], 0)

            return student_tensors, teacher_tensors

        else:
            image_tensors = [self.transform(image) for image in batch]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

            return image_tensors, image_tensors


class AlignCollate_SelfSL(object):
    def __init__(self, opt):
        self.opt = opt
        self.transform = ResizeNormalize((opt.imgW, opt.imgH))
        if "MoCo" in opt.self:
            self.MoCo_augment = MoCo_augment(opt)

    def __call__(self, batch):

        if "RotNet" in self.opt.self:
            rotate_images = []
            rotate_labels = []
            for image in batch:
                image_rotated_0 = image
                image_rotated_90 = image.transpose(PIL.Image.ROTATE_90)
                image_rotated_180 = image.transpose(PIL.Image.ROTATE_180)
                image_rotated_270 = image.transpose(PIL.Image.ROTATE_270)
                rotate_images.extend(
                    [
                        image_rotated_0,
                        image_rotated_90,
                        image_rotated_180,
                        image_rotated_270,
                    ]
                )
                rotate_labels.extend(
                    [0, 1, 2, 3]
                )  # corresponds to 0, 90, 180, 270 degrees, respectively.

            image_tensors = [self.transform(image) for image in rotate_images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

            return image_tensors, rotate_labels

        elif "MoCo" in self.opt.self:
            q_list = []
            k_list = []
            for image in batch:
                q, k = self.MoCo_augment(image)
                q_list.append(q)
                k_list.append(k)

            q_tensors = torch.cat([t.unsqueeze(0) for t in q_list], 0)
            k_tensors = torch.cat([t.unsqueeze(0) for t in k_list], 0)

            return q_tensors, k_tensors


# from https://github.com/facebookresearch/moco
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, image):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        image = image.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))
        return image


class RandomCrop(object):
    """RandomCrop,
    RandomResizedCrop of PyTorch 1.6 and torchvision 0.7.0 work weird with scale 0.90-1.0.
    i.e. you can not always make 90%~100% cropped image scale 0.90-1.0, you will get central cropped image instead.
    so we made RandomCrop (keeping aspect ratio version) then use Resize.
    """

    def __init__(self, scale=[1, 1]):
        self.scale = scale

    def __call__(self, image):
        width, height = image.size
        crop_ratio = random.uniform(self.scale[0], self.scale[1])
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)

        x_start = random.randint(0, width - crop_width)
        y_start = random.randint(0, height - crop_height)
        image_crop = image.crop(
            (x_start, y_start, x_start + crop_width, y_start + crop_height)
        )
        return image_crop


class Text_augment(object):
    """Augmentation for Text recognition"""

    def __init__(self, opt):
        self.opt = opt
        augmentation = []
        aug_list = self.opt.Aug.split("-")
        for aug in aug_list:
            if aug.startswith("Blur"):
                maximum = float(aug.strip("Blur"))
                augmentation.append(
                    transforms.RandomApply([GaussianBlur([0.1, maximum])], p=0.5)
                )

            if aug.startswith("Crop"):
                crop_scale = float(aug.strip("Crop")) / 100
                augmentation.append(RandomCrop(scale=(crop_scale, 1.0)))

            if aug.startswith("Rot"):
                degree = int(aug.strip("Rot"))
                augmentation.append(
                    transforms.RandomRotation(
                        degree, resample=PIL.Image.BICUBIC, expand=True
                    )
                )

        augmentation.append(
            transforms.Resize(
                (self.opt.imgH, self.opt.imgW), interpolation=PIL.Image.BICUBIC
            )
        )
        augmentation.append(transforms.ToTensor())
        self.Augment = transforms.Compose(augmentation)
        print("Use Text_augment", augmentation)

    def __call__(self, image):
        image = self.Augment(image)
        image.sub_(0.5).div_(0.5)

        return image


class MoCo_augment(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, opt):
        self.opt = opt

        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(
                (opt.imgH, opt.imgW), scale=(0.2, 1.0), interpolation=PIL.Image.BICUBIC
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]

        self.Augment = transforms.Compose(augmentation)
        print("Use MoCo_augment", augmentation)

    def __call__(self, x):
        q = self.Augment(x)
        k = self.Augment(x)
        q.sub_(0.5).div_(0.5)
        k.sub_(0.5).div_(0.5)

        return [q, k]
