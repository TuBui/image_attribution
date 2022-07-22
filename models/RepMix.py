#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import pairwise_distance
# from utils import augment
from utils import HParams, load_config, print_config
from utils.augment_imagenetc import get_transforms
from . import torch_layers as tl


def default_hparams():
    hps_base = HParams(
        # model base params
        d_embed=256,  # embedding dimension
        num_classes=8,  # num of classes, should be updated

        # loss
        mixup_samples=2,  # no. of images to mix
        mixup_beta=0.4,  # 0. if uniform
        mixup_level=5,  # position of mixup layer

        # transform settings
        transform_prob=1.0,  # transform applying probability
        do_dct_target=False,  # do DCT on image prior imagenet-c and make it to target y
        do_dct_input=False,  # do DCT on image after imagenet-c and make it to input x
        do_hier_classify=False,  # legacy
        pertubation=True,  # perturb image with imagenet-c transform
        residual=False,  # compute noise residual image
        loss_weight='term',  # term, pos

        device='cuda:0',
        inference=False,
        gpus=1,
        img_rsize=256,  # image input size
        img_csize=224,  # image crop size
        max_c=15,  # max no of imagenetc tforms for augmentation
        early_stop=True,
        img_mean=[0.5,0.5,0.5], # [0.485, 0.456, 0.406],
        img_std=[0.5,0.5,0.5],  # [0.229, 0.224, 0.225],
        nepochs=30,
        batch_size=32,
        train_nworkers=4,
        val_nworkers=2,
        optimizer='Adam',
        lr_betas=[0.9, 0.999],  # Adam lr beta values
        lr_scheduler='multistep',  # step, multistep
        lr=0.0001,  # initial lr
        lr_gamma=0.85,  # lr gamma in StepLR scheduler
        grad_clip=0.,  # gradient clipping, 0 if no clip
        grad_iters=1,  # number of iterations before loss and gradient are computed and network params are updated
        weight_decay=0.0005,
        resume=True,
        save_every=1,  # save model every x epoch
        save_topk=10,
        report_every=500,  # tensorboard report every x iterations
        val_every=2,  # validate every x epoch
        checkpoint_path='./'
    )
    return hps_base


class ResnetMixup(nn.Module):
    def __init__(self, d_embed, mix_level=0, nmix=2, beta=0.4, inference=False):
        super().__init__()
        self.inference = inference
        model = torchvision.models.resnet50(pretrained=True, progress=False)
        model.fc = nn.Linear(model.fc.in_features, d_embed)
        model = list(model.children())
        model = [nn.Sequential(*model[:4])] + model[4:-2] + [nn.Sequential(model[-2], nn.Flatten(1), model[-1])]
        assert mix_level <= len(model)
        mx_layer = tl.MixupLayer(nmix, beta)
        model.insert(mix_level, mx_layer)
        self.mix_level = mix_level
        self.model = nn.ModuleList(model)
        print(f'ResnetMixup, inference mode: {self.inference}:\n', self.model)

    def forward(self, x, ratio=None):
        for i, layer in enumerate(self.model):
            if i== self.mix_level:
                if not self.inference:  # turn on only in training/val
                    x = layer(x, ratio)
            else:
                x = layer(x)
        return x


class RepMix(pl.LightningModule):
    """
    RepMix
    """
    def __init__(self, **kwargs):
        super().__init__()
        hps = default_hparams().values()
        hps.update(**kwargs)
        self.save_hyperparameters(hps)  # save to hparams
        self.build_model()  # build model
        self.build_loss()
        self.transforms = get_transforms(self.hparams.img_mean, self.hparams.img_std, self.hparams.img_rsize, self.hparams.img_csize, self.hparams.pertubation, self.hparams.do_dct_input, self.hparams.residual, self.hparams.max_c)

    def build_model(self):
        # raise NotImplementedError
        hps = self.hparams
        kwargs = {'mix_level': hps.mixup_level, 'nmix': hps.mixup_samples, 'beta': hps.mixup_beta, 'inference': hps.inference}
        self.backbone = ResnetMixup(hps.d_embed, **kwargs)

    def build_loss(self):
        hps = self.hparams
        d_embed = hps.d_embed
        self.attribution = nn.Linear(d_embed, hps.num_classes)
        self.attr_ce = nn.CrossEntropyLoss(reduction='none')

        self.act = nn.Softmax(dim=1) # nn.Sigmoid()
        self.detection = nn.Linear(d_embed, 2)
        self.det_ce = nn.CrossEntropyLoss(torch.tensor([1, 1./(hps.num_classes-1)]), reduction='none')
    
    def forward(self, x):
        emb = self.backbone(x['x'], x['beta'])  # x['x'] \in [B,3,H,W], x['beta'] \in [B]
        out = {'embedding': emb}
        attribution = self.attribution(out['embedding'])  # [B, num_classes]
        out['detection'] = self.detection(out['embedding'])  # [B,2]
        detection = self.act(out['detection'])
        attribution0 = attribution[:,0] * detection[:,0]
        attribution1 = attribution[:,1:] * detection[:,1].unsqueeze(1)
        
        out['attribution'] = torch.cat([attribution0.unsqueeze(1), attribution1],1)
        return out 

    def compute_loss(self, pred, target):
        hps = self.hparams
        out = {}
        if hps.inference:
            out['attribution_loss'] = torch.mean(self.attr_ce(pred['attribution'], target['y_gan']))
            out['detection_loss'] = torch.mean(self.det_ce(pred['detection'], target['y_det']))
        else:
            mnum = hps.mixup_samples
            if mnum==1:
                target['y_gan'] = target['y_gan'].unsqueeze(1)
                target['beta'] = target['beta'].unsqueeze(1)
                target['y_det'] = target['y_det'].unsqueeze(1)
            y_det = target['y_det']
            out['attribution_loss'] = torch.mean(sum([self.attr_ce(pred['attribution'], target['y_gan'][:,i])*target['beta'][:,i] for i in range(mnum)]))
            out['detection_loss'] = torch.mean(sum([self.det_ce(pred['detection'], target['y_det'][:,i])*target['beta'][:,i] for i in range(mnum)]))
        out['total_loss'] = sum(list(out.values()))
        return out

    def compute_accuracy(self, pred, target):
        """
        compute standard clasification accuracy
        """
        logits = F.softmax(pred['attribution'])
        bsz = logits.shape[0]
        acc = [torch.sum(logits[i, target['y_gan'][i]]*target['beta'][i]) for i in range(bsz)]
        return sum(acc)/bsz

    def any_step(self, batch, batch_idx, stage):
        x, y = batch
        logits = self(x)
        loss_dict = self.compute_loss(logits, y)

        for key, val in loss_dict.items():
            on_epoch = True if key=='total_loss' else False
            self.log(f'{stage}_{key}', val, on_step=True, on_epoch=on_epoch)

        acc = self.compute_accuracy(logits, y)
        self.log(f'{stage}_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss_dict['total_loss']

    def on_epoch_start(self):
        print('\n')

    def training_step(self, batch, batch_idx):
        return self.any_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        return self.any_step(batch, batch_idx, stage='val')

    def test_step(self, batch, batch_idx):
        return self.any_step(batch, batch_idx, stage='test')

    def configure_optimizers(self):
        if self.hparams.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=self.hparams.lr_betas, weight_decay=5e-4)
        if self.hparams.optimizer == 'SGD':
            decay, no_decay, no_decay_names = [], [], []
            for name, param in self.named_parameters():
                if len(param.shape)==1 or name.endswith('.bias') or not param.requires_grad:
                    no_decay.append(param)
                    no_decay_names.append(name)
                else:
                    decay.append(param)
            print(f'SGD with weight decay on {len(decay)}/{len(decay)+len(no_decay)} weight tensors.') 
            params = [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': 5e-4}]
            optimizer = torch.optim.SGD(params, lr=self.hparams.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=self.hparams.lr_gamma)  # gamma=0.8 reduce 10x every 10 epochs; choose in [0.75,0.8,0.85]
        return [optimizer], [scheduler]
