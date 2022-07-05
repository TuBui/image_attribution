#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
defining augment classes that can be used as a pytorch transform
@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np 
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torchvision.transforms import transforms


__all__ = ['RandomJpegCompression', 'RandomGaussianBlur','RandomRecompression', 'RandomGaussianNoise', 'RandomPatchShuffle']


class RandomJpegCompression(object):
    def __init__(self, min_val=30, max_val=100, p=1.):
        self.compress_range = np.arange(min_val, max_val)
        self.p = p

    def __call__(self, x):
        if np.random.rand() >= self.p:
            return x 
        output = BytesIO()
        c = int(np.random.choice(self.compress_range))
        x.save(output, 'JPEG', quality=c)
        x = Image.open(output)
        return x 

    def __repr__(self):
        s = f'(min={self.compress_range[0]}, max={self.compress_range[-1]}, p={self.p})'
        return self.__class__.__name__ + s


class RandomRecompression(object):
    """
    re-compress image to smaller size and scale back
    """
    def __init__(self, max_scale=4, p=1.):
        self.compress_range = np.arange(int(1/max_scale*1000), 1000)
        self.p = p

    def __call__(self, x):
        if np.random.rand() >= self.p:
            return x 
        c = int(np.random.choice(self.compress_range))
        h,w = x.height, x.width
        nh, nw = int(h*c/1000), int(w*c/1000)
        x = x.resize((nw, nh), Image.BILINEAR).resize((w,h), Image.BILINEAR)
        return x 

    def __repr__(self):
        s = f'(min={self.compress_range[0]}, max={self.compress_range[-1]}, p={self.p})'
        return self.__class__.__name__ + s


class RandomGaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size=10, p=1.):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()
        self.p = p

    def __call__(self, img):
        if np.random.rand() >= self.p:
            return img
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

    def __repr__(self):
        s = f'(kernel_size={self.k}, sigma=[0.1, 2.0], p={self.p})'
        return self.__class__.__name__ + s


class RandomGaussianNoise(object):
    """additive gaussian noise"""
    def __init__(self, noise_range=[.08, .12, 0.18, 0.26, 0.38], p=1.):
        self.range = noise_range
        self.p = p

    def __call__(self, x):
        if np.random.rand() >= self.p:
            return x 
        c = np.random.choice(self.range)
        x = np.array(x) / 255.
        x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255 
        return Image.fromarray(x.astype(np.uint8))

    def __repr__(self):
        s = f'(scale={self.range}, p={self.p})'
        return self.__class__.__name__ + s


class RandomPatchShuffle(object):
    def __init__(self, patch_size=16):
        self.ps = patch_size

    def __call__(self, x):
        h, w = x.height, x.width
        assert h==w and h % self.ps==0
        npatches = h // self.ps
        x = np.array(x).reshape(npatches, self.ps, npatches, self.ps, 3)
        x = x.transpose(0,2,1,3,4).reshape(-1, self.ps, self.ps, 3)
        np.random.shuffle(x)
        x = x.reshape(npatches, npatches, self.ps, self.ps, 3).transpose(0,2,1,3,4)
        x = x.reshape(h,w,3)
        return Image.fromarray(x)

    def __repr__(self):
        s = f'(patch_size={self.ps})'
        return self.__class__.__name__ + s