#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:02:52 2023

@author: teemu
"""

import numpy as np

def cross_validation_index(
    N, 
    nfolds, 
    random_seed = None, 
    labs = None, 
    stratified = False, 
    anti_stratified = False
):
    rng = np.random.default_rng(random_seed)
    u,n = np.unique(labs, return_counts = True)
    if anti_stratified:
        '''Stratified cv folds such that each fold has different studies in holdout
           Procedure: randomly order studies and samples within studies, 
           then cut in equal slices'''
        b = rng.permutation(np.arange(n.shape[0]))
        c = np.cumsum(n[b])#[np.argsort(b)]
        a = [rng.permutation(np.arange(n[b[i]])) + 
             (c[i-1] if i else 0) 
             for i in np.arange(n.shape[0])]
        d = np.full((N,), np.nan)
        for j in np.arange(n.shape[0]):
            d[labs == u[b[j]]] = a[j]
        cv_ind = d // -(N // -nfolds) # floor of ceiling of integer division
    elif stratified:
        '''Homogenous cv folds such that each fold has samples from every study'''
        a = [rng.permutation(np.arange(n[i])) + 
             (np.cumsum(n)[i-1] if i else 0) for i in np.arange(n.shape[0])]
        cv_ind = np.full((N,), np.nan)
        for j in np.arange(n.shape[0]):
            cv_ind[labs == u[j]] = a[j] % nfolds
    else:
        '''Completely random cv fold assignment'''
        cv_ind = rng.choice(np.arange(N), N, replace = False) % nfolds
    return cv_ind