import os
import sys
import argparse
import time
import numpy as np
from tqdm import tqdm
import random
import logging
import cv2

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
            'use_abi': config.dataset_use_abi,
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
    train_ds = _get_dataset(dataset_class, config.dataset_train_roots, True, config, **ds_kwargs)
    valid_ds = _get_dataset(dataset_class, config.dataset_valid_roots, False, config, **ds_kwargs)
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.dataset_train_batch_size,
        shuffle=True,
        num_workers=config.dataset_num_workers,
        collate_fn=collate_fn_filter_none,
        pin_memory=config.dataset_pin_memory,
        drop_last=False,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=config.dataset_valid_batch_size,
        shuffle=True,
        num_workers=config.dataset_num_workers,
        collate_fn=collate_fn_filter_none,
        pin_memory=config.dataset_pin_memory,
        drop_last=True,
    )
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
    return train_dataloader, valid_dataloader, test_dataloaders


if __name__ == "__main__":
    config = _parse_arguments()
    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    _set_random_seed(config.global_seed)
    logging.info(config)
    os.makedirs(f"./saved_models/{config.global_name}", exist_ok=True)
    os.makedirs(f"./tensorboard", exist_ok=True)
    config.writer = SummaryWriter(log_dir=f"./tensorboard/{config.global_name}")

    """dataset preparation"""
    logging.info('Construct dataset.')
    train_dataloader, valid_dataloader, test_dataloaders = _get_databaunch(config)
    train_data_loader_iter = iter(train_dataloader)
    config.iter_num = len(train_data_loader_iter)
    logging.info(f"each epoch iteration: {config.iter_num}")

    model = DINO_Finetune(config)

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if config.model_pretrain_checkpoint:
        logging.info(f'Read pretrain vision model from {config.model_pretrain_checkpoint}.')
        pretrained_state_dict = torch.load(config.model_pretrain_checkpoint)
        dd = model.state_dict()
        for name in dd.keys():
            try:
                dd[name] = pretrained_state_dict['teacher'][name]
            except:
                logging.info(f'{name}')
        model.load_state_dict(dd)
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

    """ setup loss """
    train_loss_avg = Averager()

    # filter that only require gradient descent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    logging.info(f"Trainable params num: {sum(params_num)}\n")

    # setup optimizer
    params_groups = utils.get_params_groups(model)
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups, lr=config.lr,
                                      betas=(0.9, 0.999), weight_decay=config.weight_decay)  # to use with ViTs

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        config.lr,  # linear scaling rule
        config.min_lr,
        config.training_epochs, len(train_dataloader),
        warmup_epochs=config.warmup_epochs,
    )

    # ============ optionally resume training ... ============
    to_restore = {'iteration': 0}
    if config.global_stage == 'pretrain-vision':
        utils.restart_from_checkpoint(
            os.path.join(config.output_dir, config.global_name, "checkpoint.pth"),
            run_variables=to_restore,
            model=model,
            optimizer=optimizer,
        )
    elif config.global_stage == 'train-supervised':
        try:
            utils.restart_from_checkpoint(
                config.model_checkpoint,
                run_variables=to_restore,
                model=model,
                optimizer=optimizer,
            )
        except:
            print('no checkpoint need load!')
    iteration = to_restore["iteration"]
    print(f'continue to train:{iteration}')
    start_time = time.time()
    """ start training """
    best_accuracy = 0.
    # training loop
    for iteration in tqdm(
            range(iteration, int(config.training_epochs * config.iter_num)),
            total=int(config.training_epochs * config.iter_num),
            position=0,
            leave=True,
    ):
        try:
            image_tensors, label_tensors = train_data_loader_iter.next()
        except StopIteration:
            train_data_loader_iter = iter(train_dataloader)
            image_tensors, label_tensors = train_data_loader_iter.next()
        except ValueError:
            print('error' * 100)
            pass

        image_tensors = image_tensors.to(device)
        label_tensors = label_tensors.squeeze().to(device)

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[iteration]

        losses, attn = model(image_tensors, label_tensors, return_loss=True)
        loss = losses.mean()
        model.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)  # gradient clipping with 5 (Default)
        optimizer.step()
        train_loss_avg.add(loss)

        if iteration % config.training_show_iters == 0:
            i = random.randint(0, config.dataset_train_batch_size - 1)
            config.writer.add_scalar(tag='metric/' + 'train_loss', scalar_value=train_loss_avg.val(),
                                     global_step=iteration)
            lr = optimizer.param_groups[0]["lr"]
            config.writer.add_scalar(tag='metric/' + 'lr', scalar_value=lr, global_step=iteration)
            logging.info(f'iteration:{iteration}--> train loss:{train_loss_avg.val()}')
            train_loss_avg.reset()

            image0 = image_tensors[i]
            x0 = image0.float() * torch.Tensor([0.2290, 0.2240, 0.2250]).to(image0.device)[..., None, None] + \
                 torch.Tensor([0.4850, 0.4560, 0.4060]).to(image0.device)[..., None, None]
            config.writer.add_image('Mask/Input_image', x0, iteration)

            overlaps = []
            ABI_scores = attn.mean(1).reshape(-1, 25, 8, 32)
            T = ABI_scores.shape[1]
            attn_scores = ABI_scores[i].detach().cpu().numpy()
            image_numpy = x0.detach().cpu().float().numpy()
            if image_numpy.shape[0] == 1:
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            x = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            for t in range(T):
                att_map = attn_scores[t, :, :]  # [feature_H, feature_W, 1] .reshape(32, 8).T
                # normalize mask
                att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + np.finfo(float).eps)
                att_map = cv2.resize(att_map, (128, 32))  # [H, W]
                # x = cv2.resize(x, (128, 32))
                att_map = (att_map * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_JET)  # [H, W, C]
                overlap = cv2.addWeighted(heatmap, 0.6, x, 0.4, 0, dtype=cv2.CV_32F)
                overlaps.append(overlap)
            char_segmention = vutils.make_grid(torch.Tensor(overlaps).permute(0, 3, 1, 2), normalize=True,
                                               scale_each=True, nrow=5)
            config.writer.add_image('Mask/vis_Maps', char_segmention, iteration)

        ###evaluate part
        if iteration % config.training_eval_iters == 0:
            logging.info('eval model')
            elapsed_time = time.time() - start_time
            model.eval()
            eval_acc_words = 0.
            eval_acc = 0.
            eval_data_name = \
                ["IIIT5k_3000",
                 "SVT",
                 # "IC03_860",
                 # "IC03_867",
                 # "IC13_857",
                 "IC13_1015",
                 "IC15_1811",
                 # "IC15_2077",
                 "SVTP",
                 "CUTE80",
                 "COCOText",
                 "CTW",
                 "TotalText",
                 "HOST",
                 "WOST",
                 ]
            evaluation_log = ''
            log = open(f'./saved_models/{config.global_name}/log_all_evaluation.txt', 'a')
            dashed_line = '-' * 80
            print(dashed_line)
            log.write(dashed_line + '\n')
            evaluation_log += 'iteration: {:} \n'.format(iteration)
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
                log.write(evaluation_log + '\n')
                log.close()
                if mean_loss >= best_accuracy:
                    checkpoint = {
                        'net': model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        'iteration': iteration,
                    }
                    torch.save(checkpoint, f'./saved_models/{config.global_name}/best_accuracy.pth')
                    best_accuracy = mean_loss
                config.writer.add_scalar(tag='metric/' + 'eval_acc', scalar_value=mean_loss, global_step=iteration)
            model.train()

        if iteration % config.training_save_iters == 0:
            checkpoint = {
                'net': model.state_dict(),
                "optimizer": optimizer.state_dict(),
                'iteration': iteration,
            }
            torch.save(checkpoint, f'./saved_models/{config.global_name}/{iteration}.pth')
