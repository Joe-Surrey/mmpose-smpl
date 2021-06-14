import argparse
import copy
import os
import os.path as osp
import time
import random

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist, set_random_seed
from mmcv.utils import get_git_hash
from mmpose.datasets import build_dataloader, build_dataset

from mmpose import __version__
from mmpose.apis import train_model
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.utils import collect_env, get_root_logger
from mmpose.datasets.datasets.top_down import TopDownCocoWholeBodyLazyDataset
from mmpose.datasets.datasets.top_down import TopDownCocoWholeBodyLazyDataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    #mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_posenet(cfg.model)

    #Datasets


    random.shuffle(cfg.data.train["ann_file"])
    random.shuffle(cfg.data.val["ann_file"])
    datasets = [None for item in range(len(cfg.workflow))]
    dataset_names = [None for item in range(len(cfg.workflow))]
    train_counter = 0
    val_counter = 0
    for index in range(len(datasets)):
        if cfg.workflow[index][0] == 'train':
            datasets[index] = build_dataset({
                                            'type': cfg.data.train['type'],
                                            'ann_file': cfg.data.train["ann_file"][train_counter],
                                            'img_prefix': cfg.data.train['img_prefix'],
                                            'data_cfg': cfg.data.train['data_cfg'],
                                            'pipeline': cfg.data.train['pipeline']
                                            })
            dataset_names[index] = f"train: {cfg.data.train['ann_file'][train_counter]}"
            train_counter = (train_counter + 1) % len(cfg.data.train["ann_file"])
        else:
            datasets[index] = build_dataset({
                                            'type': cfg.data.val['type'],
                                            'ann_file': cfg.data.val["ann_file"][val_counter],
                                            'img_prefix': cfg.data.val['img_prefix'],
                                            'data_cfg': cfg.data.val['data_cfg'],
                                            'pipeline': cfg.data.val['pipeline']
                                        })
            dataset_names[index] = f"val: {cfg.data.val['ann_file'][val_counter]}"
            val_counter = (val_counter + 1) % len(cfg.data.val["ann_file"])





    dataloader_setting = dict(
        samples_per_gpu=cfg.data.get('samples_per_gpu', {}),
        workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))
    print(f"Shuffle datasets: {dataloader_setting['shuffle']}")
    print(type(datasets[0]))
    if isinstance(datasets[0], TopDownCocoWholeBodyLazyDataset):
        print("topdown_coco_wholebody_lazy_dataset")
        dataloader_setting["DataLoader"] = TopDownCocoWholeBodyLazyDataloader
    else:
        print(type(datasets[0]))


    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in datasets
    ]
    for i in range(1, len(data_loaders)):
        data_loaders[i].prev = data_loaders[i - 1]
    if len(data_loaders) > 1:
        data_loaders[0].prev = data_loaders[-1]
    print("???????")
    print(os.getpid())


    for i, data_batch in enumerate(data_loaders[0]):
        print(i)
    print("DOne")
if __name__ == '__main__':
    main()
