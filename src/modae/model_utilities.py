#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:55:07 2023

@author: rintala
"""

import tensorflow as tf
from modae.data_batch import batch_to_step_args

def get_embeddings(model, dataset, tuple_dataset = True):
    z_list = []
    for batch in zip(dataset):
        step_args = batch_to_step_args(
            batch, 
            tuple_dataset = tuple_dataset)
        z_list.append(model.encode(step_args['x']))
    return tf.concat(z_list, axis = 0)
    