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
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image 


def unnormalise(tensor, mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5]):
    """
    unnormalise a tensor and return an image
    """
    im = tensor.cpu().numpy().squeeze()  # [C,H,W]
    im = im.transpose(1,2,0) * np.array(std)[None, None,...] + np.array(mean)[None, None,...]
    im = np.clip(im*255, 0, 255).astype(np.uint8)
    return Image.fromarray(im)
