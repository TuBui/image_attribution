#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import fftpack
import numpy as np 
from PIL import Image 


__all__ = ['DCT']


def log_scale(array, epsilon=1e-12):
    """Log scale the input array.
    """
    array = np.abs(array)
    array += epsilon  # no zero in log
    array = np.log(array)
    return array

def dct2(array):
    """2D DCT"""
    array = fftpack.dct(array, type=2, norm="ortho", axis=0)
    array = fftpack.dct(array, type=2, norm="ortho", axis=1)
    return array

def idct2(array):
    """inverse 2D DCT"""
    array = fftpack.idct(array, type=2, norm="ortho", axis=0)
    array = fftpack.idct(array, type=2, norm="ortho", axis=1)
    return array


class DCT(object):
    def __init__(self, log=True, uint8=True):
        self.log = log 
        self.uint8 = uint8

    def __call__(self, x):
        x = np.array(x)
        x = dct2(x)
        if self.log:
            x = log_scale(x)
        # normalize
        x = (x - x.min())/(x.max() - x.min()).astype(np.float32)
        if self.uint8:
            x = Image.fromarray(np.clip(x * 255, 0, 255).astype(np.uint8))
        return x

    def __repr__(self):
        s = f'(Discrete Cosine Transform, logarithm={self.log}, uint8={self.uint8})'
        return self.__class__.__name__ + s