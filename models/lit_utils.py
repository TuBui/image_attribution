#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch lightning utils
@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pytorch_lightning as pl
from tqdm import tqdm


class LitBar(pl.callbacks.ProgressBar):
    def __init__(self, *args, **kwargs):
        if 'refresh_rate' not in kwargs:
            kwargs['refresh_rate'] = 100  # refresh every 100 iters
        super().__init__(*args, **kwargs)
        
    def init_train_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for training. """
        bar = super().init_train_tqdm()
        bar.miniters = self.total_train_batches//10
        bar.mininterval = 60
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.main_progress_bar is not None
        bar = super().init_validation_tqdm()
        bar.miniters = self.total_val_batches//10
        bar.mininterval = 60
        bar.position = (2 * self.process_position + has_main_bar)
        bar.leave = True
        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = super().init_test_tqdm()
        bar.miniters = self.total_test_batches//10
        bar.mininterval = 60
        bar.leave = True
        return bar
