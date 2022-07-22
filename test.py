#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from torch.utils.data import DataLoader, DistributedSampler
import random
import glob
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from utils import Timer
from utils.folder import ImageFolder
from utils.analysis import to_cuda, compute_nmi
from models import RepMix


def main(args):
    model = RepMix.RepMix.load_from_checkpoint(args.weight, inference=True).cuda()
    model.eval()
    batch_size = model.hparams.batch_size

    # data
    test_set = ImageFolder(args.data_dir, args.data_list)
    test_set.set_transform(model.transforms['clean'])
    print(test_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # output
    prob = []
    label = []
    niters = len(test_loader)
    with torch.no_grad():
        for x,y in tqdm(test_loader, total=niters, miniters=niters//10, mininterval=10):
            x = to_cuda(x)
            x['beta'] = None  # beta isn't needed for inference
            y = y['y_gan'].cpu().numpy()
            out = model(x)['attribution'].cpu().numpy()
            prob.append(out)
            label.append(y)
    label = np.concatenate(label)
    prob = np.concatenate(prob)
    pred = prob.argmax(axis=1)
    pacc = np.mean((pred > 0) == (label > 0))  # real/fake prediction
    acc = np.mean(pred == label)  # attribution
    nmi = compute_nmi(pred, label)
    print(f'Detection acc, Attribution acc, NMI: {pacc:.4f} & {acc:.4f} & {nmi:.4f}')

    res = {'attr': prob, 'labels': label}
    np.savez('res.npz', **res)
    np.save('emb.npy', prob)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test pl model')
    parser.add_argument('-d', '--data_dir', default='/data/processed', help='test data directory')
    parser.add_argument('-l', '--data_list', default='/data/test.csv', help='test lst')
    parser.add_argument('-w', '--weight', default='/model/last.ckpt', help='model weight')

    args = parser.parse_args()
    timer = Timer()
    main(args)
    print('Done. Total time: %s' % timer.time(False))