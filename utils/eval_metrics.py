#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
common evaluation metrics
@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np 
from sklearn.metrics import pairwise_distances
from .helpers import ProgressBar, Timer


def pairwise_dist(array_a, array_b, metric='L2'):
    """
    compute pairwise distance between two matrices of size (m,d) and (n,d)
    output: distance matrix of size (m,n)
    """
    if metric.lower() == 'hamming':
        # out = np.bitwise_xor(array_a[:, None, :], array_b[None,...]).sum(axis=-1)
        out = pairwise_distances(array_a, array_b, 'hamming')
    else:
        q_mag = np.square(array_a).sum(axis=1)  # (m,)
        d_mag = np.square(array_b).sum(axis=1)  # (n,)
        qd = array_a.dot(array_b.T)  # (m,n)
        if metric.lower() == 'l2':
            out = q_mag[:, None] + d_mag[None, :] - 2*qd
        elif metric.lower() == 'cosine':
            out = 1 - qd / (np.sqrt(q_mag[:, None] * d_mag[None, :]) + np.finfo(np.float32).eps)
    return out


class RetrievalMetrics(object):
    def __init__(self, qfeats=None, qlabels=None, dfeats=None, dlabels=None, metric='L2'):
        self.metric = metric.lower()
        assert self.metric in ['l2', 'hamming', 'cosine'], 'Error! metric not supported.'
        self.dtype = np.bool if self.metric == 'hamming' else np.float32
        self.qfeats, self.dfeats, self.qlabels, self.dlabels = None, None, None, None
        self.register(qfeats=qfeats, qlabels=qlabels, dfeats=dfeats, dlabels=dlabels)
        
    def register(self, **kwargs):
        if 'qfeats' in kwargs and kwargs['qfeats'] is not None:
            self.qfeats = np.asarray(kwargs['qfeats']) + np.zeros((1,1), dtype=self.dtype)  # (m,d)
        if 'dfeats' in kwargs and kwargs['dfeats'] is not None:
            self.dfeats = np.asarray(kwargs['dfeats']) + np.zeros((1,1), dtype=self.dtype)  # (m,d)
        if 'qlabels' in kwargs and kwargs['qlabels'] is not None:
            self.qlabels = np.asarray(kwargs['qlabels']).squeeze()[:, None]  # (m,1)
        if 'dlabels' in kwargs and kwargs['dlabels'] is not None:
            self.dlabels = np.asarray(kwargs['dlabels']).squeeze()  # (n,)
        self.rel = None

    @staticmethod
    def _retrieve(qfeats, dfeats, metric):
        """
        perform querying qfeats against dfeats
        :return (m,n) containing id of the returned images
        """
        dist = pairwise_dist(qfeats, dfeats, metric)
        if metric.lower() == 'hamming':
            ids = dist.argsort(kind='stable').astype(np.uint32)
        else:
            ids = dist.argsort(kind='quicksort').astype(np.uint32)
        return ids

    def compute_relevance_matrix(self, qfeats, qlabels, dfeats, dlabels):
        """
        compute (m,n) binary relevance matrix 
        """
        ret_ids = self._retrieve(qfeats, dfeats, self.metric)
        ret_labels = dlabels[ret_ids]
        self.rel = ret_labels == qlabels
        self.ret_ids = ret_ids
        return self.rel

    def mAP(self, top_k=0, reuse_rel=False, chunk_size=0, return_ap=False, verbose=False):
        """
        compute retrieval mAP (scalar)
        if top_k is non-zero, compute average precision @top_k results
        reuse_rel:  reuse previously computed relevance matrix; only set to True if
            you know what u r doing.
        chunk_size: split query set into chunks (more memory efficient), 0 if query the whole set
        """
        nqueries = len(self.qlabels)
        if chunk_size:
            reuse_rel = False  # can't reuse relevance metric when chunking
        else:
            chunk_size = nqueries
        ap_list = []
        if verbose:
            prog = ProgressBar(nqueries, 10)
            timer = Timer()
        for i in range(0, nqueries, chunk_size):
            if verbose:
                showed = prog.show_progress(i)
                if showed:
                    print(f'Total time: {timer.time(False)}')
            qfeats = self.qfeats[i:min(nqueries, i + chunk_size)]
            qlabels = self.qlabels[i:min(nqueries, i + chunk_size)]

            rel = self.rel if reuse_rel else self.compute_relevance_matrix(qfeats, qlabels, self.dfeats, self.dlabels)
            if top_k:  # just mean precision
                AP = rel[:, :top_k].sum(axis=1) / float(top_k)
            else:
                P = np.cumsum(rel,axis=1) / (np.arange(rel.shape[1])[None,:] + 1.)
                AP = np.sum(P*rel,axis=1) / np.clip(rel.sum(axis=1), 1, None)  # avoid /0
            ap_list.append(AP)
        ap_list = np.concatenate(ap_list)
        mAP = ap_list.mean()
        out = (mAP, ap_list) if return_ap else mAP
        return out

    def pr_curve(self, reuse_rel=False):
        """
        compute precision and recall
        """
        rel = self.rel if reuse_rel else self.compute_relevance_matrix(self.qfeats, self.qlabels, self.dfeats, self.dlabels)
        P = np.cumsum(rel,axis=1) / (np.arange(rel.shape[1])[None,:] + 1.)
        P_ave = P.mean(axis=0)
        R = np.cumsum(rel, axis=1) / rel.sum(axis=1, keepdims=True)
        R_ave = R.mean(axis=0)
        return P_ave, R_ave
