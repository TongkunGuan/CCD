import os
import sys
import argparse
import time
import numpy as np
from tqdm import tqdm
import random
import logging
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn.init as init
import torch.utils.data
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from fastai.vision import *

from Dino.model.dino_vision import DINO_Finetune
from Dino.utils.utils import Config, Logger, MyConcatDataset
from Dino.utils.util import Averager
from Dino.dataset.dataset_pretrain import ImageDataset, collate_fn_filter_none
from Dino.dataset.datasetsupervised_kmeans import ImageDatasetSelfSupervisedKmeans
from Dino.metric.eval_acc import TextAccuracy
from Dino.modules import utils
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_random_seed(seed):
    cudnn.deterministic = True
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        logging.warning('You have chosen to seed training. '
                        'This will slow down your training!')


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('-b', '--batch_size', type=int, default=None,
                        help='batch size')
    parser.add_argument('--run_only_test', action='store_true', default=None,
                        help='flag to run only test and skip training')
    parser.add_argument('--test_root', type=str, default=None,
                        help='path to test datasets')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint')
    parser.add_argument('--vision_checkpoint', type=str, default=None,
                        help='path to vision model pretrained')
    parser.add_argument('--debug', action='store_true', default=None,
                        help='flag for running on debug without saving model checkpoints')
    parser.add_argument('--model_eval', type=str, default=None,
                        choices=['alignment', 'vision', 'language'],
                        help='model decoder that outputs predictions')
    parser.add_argument('--workdir', type=str, default=None,
                        help='path to workdir folder')
    parser.add_argument('--subworkdir', type=str, default=None,
                        help='optional prefix to workdir path')
    parser.add_argument('--epochs', type=int, default=None,
                        help='number of training epochs')
    parser.add_argument('--eval_iters', type=int, default=None,
                        help='evaluate performance on validation set every this number iterations')
    args = parser.parse_args()
    config = Config(args.config)
    if args.batch_size is not None:
        config.dataset_train_batch_size = args.batch_size
        config.dataset_valid_batch_size = args.batch_size
        config.dataset_test_batch_size = args.batch_size
    if args.run_only_test is not None:
        config.global_phase = 'Test' if args.run_only_test else 'Train'
    if args.test_root is not None:
        config.dataset_test_roots = [args.test_root]
    args_to_config_dict = {
        'checkpoint': 'model_checkpoint',
        'vision_checkpoint': 'model_vision_checkpoint',
        'debug': 'global_debug',
        'model_eval': 'model_eval',
        'workdir': 'global_workdir',
        'subworkdir': 'global_subworkdir',
        'epochs': 'training_epochs',
        'eval_iters': 'training_eval_iters',
    }
    for args_attr, config_attr in args_to_config_dict.items():
        if getattr(args, args_attr) is not None:
            setattr(config, config_attr, getattr(args, args_attr))
    return config


def _get_databaunch(config):
    def _get_dataset(ds_type, paths, is_training, config, **kwargs):
        kwargs.update({
            'img_h': config.dataset_image_height,
            'img_w': config.dataset_image_width,
            'max_length': config.decoder_max_seq_len,
            'case_sensitive': config.dataset_case_sensitive,
            'charset_path': config.dataset_charset_path,
            'data_aug': config.dataset_data_aug,
            'deteriorate_ratio': config.dataset_deteriorate_ratio,
            'multiscales': config.dataset_multiscales,
            'data_portion': config.dataset_portion,
            'filter_single_punctuation': config.dataset_filter_single_punctuation,
            'mask': config.dataset_mask,
            'type': config.dataset_charset_type,
        })
        datasets = []
        for p in paths:
            subfolders = [f.path for f in os.scandir(p) if f.is_dir()]
            if subfolders:  # Concat all subfolders
                datasets.append(_get_dataset(ds_type, subfolders, is_training, config, **kwargs))
            else:
                datasets.append(ds_type(path=p, is_training=is_training, **kwargs))
        if len(datasets) > 1:
            return MyConcatDataset(datasets)
        else:
            return datasets[0]

    bunch_kwargs = {}
    ds_kwargs = {}
    bunch_kwargs['collate_fn'] = collate_fn_filter_none
    if config.dataset_scheme == 'selfsupervised_kmeans':
        dataset_class = ImageDatasetSelfSupervisedKmeans
        if config.dataset_augmentation_severity is not None:
            ds_kwargs['augmentation_severity'] = config.dataset_augmentation_severity
        ds_kwargs['supervised_flag'] = ifnone(config.model_contrastive_supervised_flag, False)
    elif config.dataset_scheme == 'supervised':
        dataset_class = ImageDataset
    test_dataloaders = []
    for eval_root in config.dataset_test_roots:
        test_ds = _get_dataset(dataset_class, [eval_root], False, config, **ds_kwargs)
        test_dataloader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=config.dataset_test_batch_size,
            shuffle=False,
            num_workers=config.dataset_num_workers,
            collate_fn=collate_fn_filter_none,
            pin_memory=config.dataset_pin_memory,
            drop_last=False,
        )
        test_dataloaders.append(test_dataloader)
    return test_dataloaders


if __name__ == "__main__":
    config = _parse_arguments()
    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    _set_random_seed(config.global_seed)
    logging.info(config)

    """dataset preparation"""
    logging.info('Construct dataset.')
    test_dataloaders = _get_databaunch(config)

    model = DINO_Finetune(config)
    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)

    if config.model_checkpoint:
        logging.info(f'Read vision model from {config.model_checkpoint}.')
        pretrained_state_dict = torch.load(config.model_checkpoint)
        # dd = model.state_dict()
        # for name in pretrained_state_dict['model'].keys():
        #     # if 'vision' in name:
        #     dd['module.'+name] = pretrained_state_dict['model'][name]
        # model.load_state_dict(dd)
        model.load_state_dict(pretrained_state_dict['net'])
    logging.info(repr(model) + "\n")

    # filter that only require gradient descent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    logging.info(f"Trainable params num: {sum(params_num)}\n")

    ###evaluate part
    logging.info('eval model')
    model.eval()
    eval_acc_words = 0.
    eval_acc = 0.
    eval_data_name = \
        [
        "IIIT5k_3000",
        "SVT",
        "IC13_1015",
        "IC15_1811",
        "SVTP",
        "CUTE80",
         "TotalText",
         "COCOText",
         "CTW",
         "HOST",
         "WOST",
         ]
    evaluation_log = ''
    dashed_line = '-' * 80
    print(dashed_line)
    with torch.no_grad():
        for i, test_dataloader in enumerate(test_dataloaders):
            eval_script = TextAccuracy(charset_path=config.dataset_charset_path,
                                       case_sensitive=config.dataset_eval_case_sensitive,
                                       model_eval='vision')
            res = eval_script.compute(model, test_dataloader)
            eval_acc += res['cwr'] * res['words']
            eval_acc_words += res['words']
            evaluation_log += f"dataset: {eval_data_name[i]} --> word_num: {res['words']} --> accuracy: {res['cwr']:0.3f}"
            evaluation_log += '\n'
        mean_loss = eval_acc / eval_acc_words
        evaluation_log += f"total_accuracy: {mean_loss:0.3f}"
        print(evaluation_log + '\n')

