#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train and (optional) test RepMix
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['WANDB_MODE'] = 'offline'
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import wandb
from argparse import ArgumentParser
from torch.utils.data import DataLoader, DistributedSampler
import random
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from utils import Timer
from utils.folder import ImageFolder, MixupFolder3, worker_init_fn
from models import RepMix
from models.lit_utils import LitBar
from utils.analysis import analyse, to_cuda, fdratio, compute_nmi
import glob
from tqdm import tqdm
import numpy as np


def main(args):
    # start
    if args.seed >= 0:
        pl.utilities.seed.seed_everything(args.seed)
    os.makedirs(args.output, exist_ok=True)
    timer = Timer()

    # config
    hps = RepMix.default_hparams()
    if args.mparams:
        hps.parse(args.mparams)

    # dataset
    train_set = MixupFolder3(args.data_dir, args.train_list, hps=hps)
    val_set = MixupFolder3(args.data_dir, args.val_list, hps=hps)
    ncats, nsems = len(train_set.classes), len(train_set.dclasses)
    hps.num_classes = ncats  # update num classes
    print(hps)

    # logger
    logger = WandbLogger(name=args.run_name, project=args.project_name, save_dir=args.output)

    # model
    model = RepMix.RepMix(**hps.values())

    #dataloader
    train_set.set_transform(model.transforms['train'])
    val_set.set_transform(model.transforms['val'])
    print(train_set, val_set)
    train_loader = DataLoader(train_set, batch_size=hps.batch_size,
                              num_workers=hps.train_nworkers,
                              worker_init_fn=worker_init_fn, 
                              shuffle=True, collate_fn=MixupFolder3.collate_fn)
    val_loader = DataLoader(val_set, batch_size=hps.batch_size,
                          num_workers=hps.val_nworkers,
                          shuffle=False, collate_fn=MixupFolder3.collate_fn)

    # Train!
    val_loss = 'val_total_loss_epoch'
    lr_monitor = LearningRateMonitor('epoch')
    checkpoint = ModelCheckpoint(args.output,
        monitor=val_loss,
        filename='ckpt-{epoch:02d}-{val_total_loss_epoch:.2f}',
        save_top_k=hps.save_topk, save_last=True, mode='min')
    last_ckpt = os.path.join(args.output, 'last.ckpt')
    last_ckpt = last_ckpt if os.path.exists(last_ckpt) else None
    callbacks = [checkpoint, lr_monitor, LitBar()]
    if hps.early_stop:
        early_stop = EarlyStopping(val_loss, patience=3, mode='min')
        callbacks.append(early_stop)

    trainer = pl.Trainer(gpus=hps.gpus, 
                         accelerator=None if hps.gpus==1 else 'dp',
                         logger=logger,
                         log_every_n_steps=hps.report_every,
                         check_val_every_n_epoch=hps.val_every,
                         max_epochs=hps.nepochs,
                         resume_from_checkpoint=last_ckpt,
                         callbacks=callbacks)

    trainer.fit(model, train_loader, val_loader)
    msg = f'Done. Best ckpt at {checkpoint.best_model_path}. Total time: {timer.time()}.'
    print(msg)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Model args
    parser.add_argument('-mp', '--mparams', default='', help='model settings')

    # Trainer args
    parser.add_argument("-p", "--project_name", default='eccv22')
    parser.add_argument("-r", "--run_name", default='repmix')
    parser.add_argument("-o", "--output", default='./output')
    parser.add_argument("-s", "--seed", type=int, default=-1, help='random seed, -1 mean no seed')

    # data args
    parser.add_argument("-tl", "--train_list", default='data/train.csv')
    parser.add_argument("-vl", "--val_list", default='data/val.csv')
    parser.add_argument('-d', '--data_dir', default='data/')

    args = parser.parse_args()

    main(args)