#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
imagefolder loader
inspired from https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import pandas as pd 
import numpy as np
import random
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple
import torch
from torchvision import transforms
# from . import debug


def worker_init_fn(worker_id):
    # to be passed to torch.utils.data.DataLoader to fix the 
    #  random seed issue with numpy in multi-worker settings
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageFolder(torch.utils.data.Dataset):
    r"""
    Customised Image Folder class for pytorch.
    Usually accept image directory and a csv list as the input.
    Usage:
        dataset = ImageFolder(img_dir, img_list)
        dataset.set_transform(some_pytorch_transforms)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
            num_workers=4, worker_init_fn=worker_init_fn)
        for x,y in loader:
            # x and y is input and target (dict), the keys can be customised.
    """
    _repr_indent = 4
    def __init__(self, data_dir, data_list, loader=pil_loader, transform=None, target_transform=None, **kwargs):
        self.root = data_dir
        self.loader = loader
        self.set_transform(transform, target_transform)
        self.build_data(data_list, data_dir)

    def set_transform(self, transform, target_transform=None):
        self.transform, self.target_transform = transform, target_transform

    def build_data(self, data_list, data_dir=None):
        """
        Args:
            data_list    (text file) must have at least 2 fields: path and label

        This method must create an attribute self.samples containing input and target samples; and another attribute N storing the dataset size

        Optional attributes: classes (list of unique classes), group (useful for 
        metric learning), N (dataset length)
        """
        assert isinstance(data_list, str) or isinstance(data_list, pd.DataFrame)
        df = pd.read_csv(data_list) if isinstance(data_list, str) else data_list
        assert 'path' in df and 'label' in df, f'[DATA] Error! {data_list} must contains "path" and "label".'
        paths = df['path'].tolist()
        labels = np.array(df['label'].tolist())
        self.N = len(labels)
        
        self.classes, inds = np.unique(labels, return_index=True)
        # class name to class index dict
        if '/' in paths[0] and os.path.exists(os.path.join(self.root, paths[0])):  # data organized by class name
            cnames = [paths[i].split('/')[0] for i in inds]
            self.class_to_idx = {key: val for key, val in zip(cnames, self.classes)}
        # class index to all samples within that class
        self.group = {}  # group by class index
        for key in self.classes:
            self.group[key] = np.nonzero(labels==key)[0]
        # self.labels = labels

        # check if data label avai
        self.dlabels = np.array(df['dlabel'].tolist())
        self.group_d = {}
        for key in list(set(self.dlabels)):
            self.group_d[key] = np.nonzero(self.dlabels==key)[0]
        self.dclasses = np.unique(self.dlabels)

        # self.samples = [(s[0], (s[1], s[2])) for s in zip(paths, self.labels, self.dlabels)]
        self.samples = {'x': paths, 'y_gan': labels, 'y_sem': self.dlabels}

    @staticmethod
    def apply_transform(transform, x):
        if isinstance(transform, list):
            for t in transform:
                x = t(x)
        elif transform is not None:
            x = transform(x)
        return x

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            dict: (x: sample, y: target, **kwargs)
        """
        path, y_gan, y_sem = self.samples['x'][index], self.samples['y_gan'][index], self.samples['y_sem'][index]
        full_path = os.path.join(self.root, path)
        sample = self.loader(full_path)
        sample = self.apply_transform(self.transform, sample)
        y_gan = self.apply_transform(self.target_transform, y_gan)
        y_sem = self.apply_transform(self.target_transform, y_sem)

        return {'x': sample}, {'y_gan': y_gan, 'y_sem': y_sem, 'y_det': 1 if y_gan else 0}

    def __len__(self) -> int:
        # raise NotImplementedError
        return self.N 

    def __repr__(self) -> str:
        head = "\nDataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self) -> str:
        return ""


# class MixupFolder(ImageFolder):
#     _repr_indent = 4
#     def __init__(self, data_dir, data_list, loader=pil_loader, transform=None, target_transform=None, hps=None):
#         # self.mixup_beta = hps.mixup_beta
#         self.mixup_level = hps.mixup_level  # 0: post aug, -1: pre aux
#         assert self.mixup_level in [-1,0], 'This dataset class is for pre/post/no augment only. Mixuplevel must be in [0,-1]'
#         self.mixup_samples = hps.mixup_samples
#         self.mixup_same_label = hps.mixup_same_label
#         self.mixup_ratio = [hps.mixup_beta] * hps.mixup_samples
#         self.dct_target = hps.do_dct_target
#         if hps.do_dct_target:
#             from .image_tools import DCT
#             self.dct = DCT(log=True)
                    
#         self.do_hier_classify = hps.do_hier_classify
#         self.do_compound_loss = hps.do_compound_loss
#         super().__init__(data_dir, data_list, loader, transform, target_transform)


#     def __getitem__(self, index: int) -> Any:
#         ids = [index]
#         ids += np.random.choice(self.N, self.mixup_samples-1).tolist()
#         x , y_gans, y_sems = [], [], []
#         for i in ids:
#             path, y_gan, y_sem = self.samples['x'][i], self.samples['y_gan'][i], self.samples['y_sem'][i]
#             full_path = os.path.join(self.root, path)
#             sample = self.loader(full_path)
#             if self.target_transform is not None:
#                 y_gan = self.target_transform(y_gan)
#                 y_sem = self.target_transform(y_sem)
#             y_gans.append(y_gan)
#             y_sems.append(y_sem)
#             x.append(sample)
#         # transform & mixup x
#         beta = np.random.dirichlet(self.mixup_ratio)
#         beta_t = torch.tensor(beta[:,None,None,None])
#         if self.transform is not None:
#             if isinstance(self.transform, list):
#                 x = [self.transform[0](x_) for x_ in x]
#                 if self.mixup_level==0:
#                     x = [self.transform[1](x_) for x_ in x]
#                     x = [self.transform[2](x_) for x_ in x]
#                     x = torch.sum(torch.stack(x)*beta_t, dim=0).float()
#                 else:  # preaug
#                     x = np.array([np.array(x_) for x_ in x])
#                     x = (x*beta[:,None,None,None]).sum(axis=0)
#                     x = Image.fromarray(np.uint8(x))
#                     x = self.transform[2](self.transform[1](x))
#             else:
#                 if self.mixup_level==0:
#                     x = [self.transform(x_) for x_ in x]
#                     x = torch.sum(torch.stack(x)*beta_t, dim=0).float()
#                 else:
#                     x = np.array([np.array(x_) for x_ in x])
#                     x = (x*beta[:,None,None,None]).sum(axis=0)
#                     x = Image.fromarray(np.uint8(x))
#                     x = self.transform(x)
#         x = {'x': x, 'beta': beta}
#         y_gan = np.array(y_gans)  # gan 
#         y_sem = np.array(y_sems)  # sem 
#         y_out = {'y_gan': y_gan, 'y_sem': y_sem, 'y_det': np.int64(y_gan > 0), 'beta': beta}
#         return x, y_out 

#     @staticmethod
#     def collate_fn(batch):
#         # batch is a list of (x,y)
#         x = {}
#         for key in batch[0][0].keys():
#             val = torch.stack([torch.as_tensor(b[0][key]) for b in batch])
#             x[key] = val 

#         y = {}
#         for key in batch[0][1].keys():
#             val = torch.stack([torch.as_tensor(b[1][key]) for b in batch])
#             y[key] = val
#         return x, y 


class MixupFolder3(ImageFolder):
    """
    return image separately (for CNN mixup and CNNTrueMixup)
    """
    _repr_indent = 4
    def __init__(self, data_dir, data_list, loader=pil_loader, transform=None, target_transform=None, hps=None):
        self.mixup_samples = hps.mixup_samples
        self.mixup_ratio = [hps.mixup_beta] * hps.mixup_samples
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(torch.tensor(self.mixup_ratio))
        super().__init__(data_dir, data_list, loader, transform, target_transform)

    def __getitem__(self, index):
        ids = [index] + np.random.choice(self.N, self.mixup_samples-1).tolist()
        x , y_gans, y_sems = [], [], []
        for i in ids:
            path, y_gan, y_sem = self.samples['x'][i], self.samples['y_gan'][i], self.samples['y_sem'][i]
            full_path = os.path.join(self.root, path)
            sample = self.loader(full_path)
            if self.transform is not None:
                if isinstance(self.transform, list):
                    x_pre = self.transform[0](sample)
                    x_post = self.transform[1](x_pre)
                    x.append(self.transform[2](x_post))
                else:
                    x.append(self.transform(sample))
            if self.target_transform is not None:
                y_gan = self.target_transform(y_gan)
                y_sem = self.target_transform(y_sem)
            y_gans.append(y_gan)
            y_sems.append(y_sem)
        x = {'x': torch.stack(x)}  # (n,c,h,w)
        x['beta'] = self.dirichlet.sample()
        y_gan = np.array(y_gans)  # gan 
        y_sem = np.array(y_sems)  # sem 
        y_out = {'y_gan': y_gan, 'y_sem': y_sem, 'y_det': np.int64(y_gan > 0), 'beta': x['beta'].clone()}
        
        return x, y_out   

    @staticmethod
    def collate_fn(batch):
        # batch is a list of (x,y)
        x = {}
        for key in batch[0][0].keys():
            if key=='x':
                val = torch.cat([b[0]['x'] for b in batch])
            else:
                val = torch.stack([torch.as_tensor(b[0][key]) for b in batch])
            x[key] = val 

        y = {}
        for key in batch[0][1].keys():
            val = torch.stack([torch.as_tensor(b[1][key]) for b in batch])
            y[key] = val
        return x, y 
