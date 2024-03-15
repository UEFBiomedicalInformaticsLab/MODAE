#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:07:49 2023

@author: teemu
"""

import numpy as np
import tensorflow as tf

def tuple_dataset_combiner(ds):
    x = []
    y = []
    y_mask = []
    times = []
    events = []
    smask = []
    covariates = []
    b = []
    dr = []
    dr_mask = []
    for dataset_index in np.arange(len(ds)):
        xi, yi, timesi, eventsi, covariatesi, bi, dri = ds[dataset_index]
        if yi is not None:
            y_maski = yi >= 0
        else:
            y_maski = None
        if times is not None:
            survival_maski = timesi >= 0
        else:
            survival_maski = None
        if dr is not None:
            dr_maski = dri >= 0
        else:
            dr_maski = None
        x.append(xi)
        y.append(yi)
        y_mask.append(y_maski)
        times.append(timesi)
        events.append(eventsi)
        smask.append(survival_maski)
        covariates.append(covariatesi)
        if len(ds) == 1:
            b.append(bi)
        else:
            b.append(tf.constant(np.full((xi.shape[0],), dataset_index, dtype = 'int64')))
        dr.append(dri)
        dr_mask.append(dr_maski)
    x = np.concatenate(x, axis = 0)
    y = np.concatenate(y, axis = 0)
    y_mask = np.concatenate(y_mask, axis = 0)
    times = np.concatenate(times, axis = 0)
    events = np.concatenate(events, axis = 0)
    smask = np.concatenate(smask, axis = 0)
    covariates = np.concatenate(covariates, axis = 0)
    b = np.concatenate(b, axis = 0)
    dr = np.concatenate(dr, axis = 0)
    dr_mask = np.concatenate(dr_mask, axis = 0)
    return {'x' : x, 
            'y' : y, 
            'y_mask' : y_mask, 
            'survival_times' : times, 
            'survival_events' : events, 
            'survival_mask' : smask, 
            'survival_covariates' : covariates, 
            'b' : b, 
            'dr' : dr,
            'dr_mask' : dr_mask}

# TODO: implement a solution for full model training without patient_and_cl
def batch_to_step_args(
        batch, 
        optimizer = None, 
        tuple_dataset = True, 
        patient_and_cl = False, 
        batch_patient_indicator = None, 
        batches = False, 
        classes = False, 
        survival = False, 
        drugs = False, 
        return_z = False):
    out = {}
    if tuple_dataset:
        out = tuple_dataset_combiner(batch)
        if not batches:
            out.pop('b', None)
        if not classes:
            out.pop('y', None)
            out.pop('y_mask', None)
        if not survival:
            out.pop('survival_times', None)
            out.pop('survival_events', None)
            out.pop('survival_mask', None)
            out.pop('survival_covariates', None)
        if not drugs:
            out.pop('dr', None)
            out.pop('dr_mask', None)
    elif patient_and_cl:
        if batch_patient_indicator is None:
            raise ValueError('batch_patient_indicator must be defined when training data includes CL and patients.')
        # Assume that all patients have class and survival data if any do 
        # and that all cell-lines have drug-response data if any do. 
        cled = np.logical_not(batch_patient_indicator)
        out['x_patient'] = tf.concat([i.get('exp') for i, cl in zip(batch, cled) if not cl], axis = 0)
        out['b_patient'] = tf.concat([i.get('batch') for i, cl in zip(batch, cled) if not cl], axis = 0)
        #b_patient = tf.concat([tf.constant(i, dtype = 'int64', shape = (d.get('exp').shape[0],)) for i, d, cl in zip(np.arange(len(batch)), batch, cled) if not cl], axis = 0)
        out['x_cl'] = tf.concat([i.get('exp') for i, cl in zip(batch, cled) if cl], axis = 0)
        out['b_cl'] = tf.concat([i.get('batch') for i, cl in zip(batch, cled) if cl], axis = 0)
        #b_cl = tf.concat([tf.constant(i, dtype = 'int64', shape = (d.get('exp').shape[0],)) for i, d, cl in zip(np.arange(len(batch)), batch, cled) if cl], axis = 0)
        out['b'] = tf.concat([out['b_patient'], out['b_cl']], axis = 0)
        #out['b_patient'] = b_patient
        #out['b_cl'] = b_cl
        classed = np.any([np.isin('class', list(i.keys())) for i in batch])
        survivaled = np.any([np.isin('time', list(i.keys())) for i in batch])
        drugged = np.any([np.isin('dr', list(i.keys())) for i in batch])
        if classes and classed:
            out['y'] = tf.concat([i.get('class') for i, cl in zip(batch, cled) if not cl], axis = 0)
            out['y_mask'] = tf.concat([i.get('cmask') for i, cl in zip(batch, cled) if not cl], axis = 0)
        if survival and survivaled:
            out['survival_times'] = tf.concat([i.get('time') for i, cl in zip(batch, cled) if not cl], axis = 0)
            out['survival_events'] = tf.concat([i.get('event') for i, cl in zip(batch, cled) if not cl], axis = 0)
            out['survival_mask'] = tf.concat([i.get('smask') for i, cl in zip(batch, cled) if not cl], axis = 0)
            covariated = np.array([np.isin('covar', list(i.keys())) for i, cl in zip(batch, cled) if not cl])
            if np.any(covariated) and np.any(np.logical_not(covariated)):
                raise ValueError('Either all or none of the survival datasets must have covariates. please use imputation.')
            if np.any(covariated):
                out['survival_covariates'] = tf.concat([i.get('covar') for i, cl in zip(batch, cled) if not cl], axis = 0)
        if drugs and drugged:
            out['dr'] = tf.concat([i.get('dr') for i, cl in zip(batch, cled) if cl], axis = 0)
            out['dr_mask'] = tf.concat([i.get('drmask') for i, cl in zip(batch, cled) if cl], axis = 0)
    elif classes:
        classed = np.array([np.isin('class', list(i.keys())) for i in batch])
        if np.any(classed):
            out['x'] = tf.concat([i.get('exp') for i, c in zip(batch, classed) if c], axis = 0)
            out['y'] = tf.concat([i.get('class') for i, c in zip(batch, classed) if c], axis = 0)
            out['y_mask'] = tf.concat([i.get('cmask') for i, c in zip(batch, classed) if c], axis = 0)
    elif survival:
        survivaled = np.array([np.isin('time', list(i.keys())) for i in batch])
        if np.any(survivaled):
            out['x'] = tf.concat([i.get('exp') for i, s in zip(batch, survivaled) if s], axis = 0)
            out['survival_times'] = tf.concat([i.get('time') for i, s in zip(batch, survivaled) if s], axis = 0)
            out['survival_events'] = tf.concat([i.get('event') for i, s in zip(batch, survivaled) if s], axis = 0)
            out['survival_mask'] = tf.concat([i.get('smask') for i, s in zip(batch, survivaled) if s], axis = 0)
            covariated = np.array([np.isin('covar', list(i.keys())) for i in batch])
            if np.any(survivaled != covariated):
                raise ValueError('Either all or none of the survival datasets must have covariates. please use imputation.')
            if np.any(covariated):
                out['survival_covariates'] = tf.concat([i.get('covar') for i, s in zip(batch, survivaled) if s], axis = 0)
    elif batches:
        out['x'] = tf.concat([i.get('exp') for i in batch], axis = 0)
        out['b'] = tf.concat([i.get('batch') for i in batch], axis = 0)
        #out['b'] = tf.concat([tf.constant(i, dtype = 'int64', shape = (d.get('exp').shape[0],)) for i,d in enumerate(batch)], axis = 0)
    elif drugs:
        drugged = np.array([np.isin('dr', list(i.keys())) for i in batch])
        if np.any(drugged):
            out['x'] = tf.concat([i.get('exp') for i, c in zip(batch, drugged) if c], axis = 0)
            out['dr'] = tf.concat([i.get('dr') for i, c in zip(batch, drugged) if c], axis = 0)
            out['dr_mask'] = tf.concat([i.get('drmask')  for i, c in zip(batch, drugged) if c], axis = 0)
    else:
        out['x'] = tf.concat([i.get('exp') for i in batch], axis = 0)
    out['return_z'] = return_z
    out['optimizer'] = optimizer
    return out