# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 2019

@author: rintala
"""

# External imports
import tensorflow as tf
import numpy as np
import pandas as pd
from copy import copy
from multiprocessing import Process

import glob
import os
import time
import sys

# Internal imports
from modae.autoencoder_model import MODAE
from modae.data_utilities import (
    parse_serialized_dataset_from_file, 
    process_patient_data, # here for compatibility with old code
    prepare_tf_dataset_from_numpy, 
    dataset_batch_setup, 
    JSONFeatureSpecDecoder
)
from modae.data_batch import batch_to_step_args
from modae.evaluation import (
    get_model_losses, 
    print_diagnostics,
    evaluate_results
)
from modae.evaluation_utilities import cross_validation_index

'''
Train a given Variational Autoencoder with given data, model and optimizer
'''
def train(
    train_dataset = None, 
    valid_dataset = None, 
    train_dataset2 = None,
    valid_dataset2 = None,
    model = None, 
    optimizer = None, 
    adversarial_optimizer = None,
    epochs = 100, 
    verbose = True, 
    print_period = 10, 
    return_losses = False, 
    early_stopping = False, 
    patience = 10, 
    train_ae_only = False, 
    train_classifier_only = False, 
    train_survival_only = False, 
    train_batch_detector_only = False, 
    train_batch_correction_only = False, 
    train_drugs_only = False, 
    supervised_metric = 'cross_entropy', 
    tuple_dataset = True,
    patient_data = False, 
    cell_line_data = False, 
    batch_patient_indicator = None, 
    batch_adversarial_iterations = 5,
    debug = False
):
    if print_period > epochs:
        print_period = epochs
    
    # Deal with possible missingness of arguments
    if valid_dataset is None:
        valid_dataset = train_dataset
    if valid_dataset2 is None:
        valid_dataset2 = train_dataset2
    
    if early_stopping:
        best_loss = np.Inf
        disappointments = 0
    
    if return_losses:
        train_losses = [pd.DataFrame({})]
        valid_losses = [pd.DataFrame({})]
    
    # Initialize (used if error is nan or always increases, e.g. in case of a bug ...)
    best_model_weights = copy(model.get_weights())
    
    train_full_model = not np.any([
        train_ae_only, 
        train_classifier_only, 
        train_survival_only, 
        train_batch_detector_only, 
        train_batch_correction_only, 
        train_drugs_only])
    if not train_batch_detector_only:
        train_step = model.train_step_gen(
            optimizer, 
            ae = train_ae_only or train_batch_correction_only or train_full_model, 
            classifier = train_classifier_only or train_full_model,
            survival = train_survival_only or train_full_model, 
            batch_detector = False,
            batch_correction = train_batch_correction_only or train_full_model, 
            drug_resp = train_drugs_only or train_full_model, 
            patient_data = patient_data and train_full_model, 
            cell_line_data = cell_line_data and train_full_model, 
            return_z = True)
    if (train_batch_detector_only or train_batch_correction_only or train_full_model) and model.batch_adversarial_model:
        adversarial_train_step = model.train_step_gen(
            adversarial_optimizer, 
            batch_detector = True,
            patient_data = patient_data and train_full_model, 
            cell_line_data = cell_line_data and train_full_model, 
            return_z = True)
    # This check is used to flag data with disjoint observations, i.e., drug-response 
    # for cell-lines only and survival and cancer-subtype for patients only. 
    patient_and_cl = train_full_model and (patient_data or cell_line_data)
    
    nan_counter = 0
    for epoch in range(1, epochs + 1):
        if verbose:
            start_time = time.time()
        # Training for one epoch
        nan_counter = 0
        if not (train_dataset2 is None):
            iterator = tf.data.Dataset.zip((train_dataset, train_dataset2))
        else:
            iterator = tf.data.Dataset.zip((train_dataset, ))
        # Actual training loop
        if not train_batch_detector_only:
            for batch in iterator:
                step_args = batch_to_step_args(
                    batch, 
                    tuple_dataset = tuple_dataset, 
                    patient_and_cl = patient_and_cl, 
                    batch_patient_indicator = batch_patient_indicator, 
                    batches = (model.batch_adversarial_model and (train_full_model or train_batch_correction_only)) or (model.deconfounder_model and train_ae_only), 
                    classes = model.supervised and (train_full_model or train_classifier_only), 
                    survival = model.survival_model and (train_full_model or train_survival_only), 
                    drugs = model.drug_response_model and (train_full_model or train_drugs_only))
                if debug:
                    step_arg_tensor_shapes = dict([(k, i.shape) for k,i in step_args.items() if isinstance(i, tf.Tensor)])
                    for k,i in step_arg_tensor_shapes.items():
                        print('Training step args \'' + str(k) + '\' shape: ' + str(i), file = sys.stderr)
                z = train_step(model, **step_args)
                if tf.math.reduce_any(tf.math.is_nan(z)):
                    nan_counter += 1
        # Adversarial training loop
        if (train_batch_detector_only or train_batch_correction_only or train_full_model) and model.batch_adversarial_model:
            for batch in iterator:
                for ba_iteration in np.arange(batch_adversarial_iterations):
                    adversarial_step_args = batch_to_step_args(
                        batch, 
                        tuple_dataset = tuple_dataset, 
                        patient_and_cl = patient_and_cl, 
                        batch_patient_indicator = batch_patient_indicator, 
                        batches = True)
                    z2 = adversarial_train_step(model, **adversarial_step_args)
                    if tf.math.reduce_any(tf.math.is_nan(z2)):
                        nan_counter += 1
        if verbose:
            end_time = time.time()
        
        if ((verbose or return_losses) and epoch % print_period == 0) or early_stopping:
            metrics_valid = get_model_losses(
                model, 
                dataset = valid_dataset,
                dataset2 = valid_dataset2,
                tuple_dataset = tuple_dataset,
                patient_and_cl = patient_and_cl, 
                get_metrics = supervised_metric != 'cross_entropy', 
                debug = debug)
            # Print validation set metrics
            if verbose and epoch % print_period == 0:
                print_diagnostics(model = model, 
                                  metrics = metrics_valid, 
                                  epoch = epoch, 
                                  start_time = start_time,
                                  end_time = end_time, 
                                  supervised_metric = supervised_metric, 
                                  set_name = 'validation set')
            if return_losses:
                metrics_train = get_model_losses(
                    model, 
                    dataset = train_dataset,
                    dataset2 = train_dataset2, 
                    tuple_dataset = tuple_dataset, 
                    patient_and_cl = patient_and_cl, 
                    get_metrics = supervised_metric != 'cross_entropy', 
                    debug = debug)
                train_losses.append(pd.DataFrame(metrics_train, index = [0]))
                valid_losses.append(pd.DataFrame(metrics_valid, index = [0]))
            if early_stopping:
                if train_ae_only:
                    total_loss = metrics_valid['reconstruction_loss_dataset1'] * model.objective_weights[0]
                    if train_dataset2 is not None:
                        total_loss += metrics_valid['reconstruction_loss_dataset2'] * model.objective_weights[0]
                elif train_classifier_only:
                    total_loss = metrics_valid[supervised_metric] * model.supervised_metric_signs[supervised_metric] * model.objective_weights[1]
                elif train_survival_only:
                    total_loss = -1. * metrics_valid['survival_log_likelihood'] * model.objective_weights[2]
                elif train_batch_detector_only:
                    # We want to maximize batch detector performance for adversarial model
                    total_loss = metrics_valid['batch_cross_entropy'] * model.objective_weights[3]
                elif train_drugs_only:
                    total_loss = metrics_valid['drug_response_mse'] * model.objective_weights[4]
                else:
                    total_loss = metrics_valid['reconstruction_loss'] * model.objective_weights[0]
                    if model.supervised:
                        total_loss += metrics_valid[supervised_metric] * model.supervised_metric_signs[supervised_metric] * model.objective_weights[1]
                    if model.survival_model:
                        total_loss -= metrics_valid['survival_log_likelihood'] * model.objective_weights[2]
                    if model.batch_adversarial_model:
                        # We want to minimize batch-effects in joint training
                        total_loss += metrics_valid['batch_dsc'] * model.objective_weights[3] 
                    if model.drug_response_model:
                        total_loss += metrics_valid['drug_response_mse'] * model.objective_weights[4]
                
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_losses = metrics_valid
                    best_model_weights = copy(model.get_weights())
                    disappointments = 0
                elif disappointments < patience:
                    disappointments += 1
                else:
                    if verbose:
                        print('Stopped at epoch ' + str(epoch - disappointments) + ' due to ' + str(patience) + 'consecutive decreases in total loss excluding regularization.')
                        print_diagnostics(model = model, 
                                          metrics = best_losses, 
                                          epoch = epoch - disappointments, 
                                          start_time = start_time,
                                          end_time = end_time, 
                                          supervised_metric = supervised_metric, 
                                          set_name = 'validation set')
                    break
            elif nan_counter == 0:
                # Save model weights at every successfull iteration
                best_model_weights = copy(model.get_weights())
    if early_stopping or nan_counter > 0:
        model.set_weights(best_model_weights)
    if return_losses:
        train_losses = pd.concat(train_losses, axis = 0)
        valid_losses = pd.concat(valid_losses, axis = 0)
        train_losses['iteration'] = np.arange(train_losses.shape[0]) + 1
        valid_losses['iteration'] = np.arange(valid_losses.shape[0]) + 1
    else:
        train_losses = None
        valid_losses = None
    return (train_losses, valid_losses)

def train_aecl_with_pretraining(
    x = None,
    y = None,
    events = None, 
    times = None, 
    covariates = None, 
    b = None, 
    dr = None,
    patient_serialized_dataset = None,
    patient_serialized_test_dataset = None,
    cl_serialized_dataset = None,
    cl_serialized_test_dataset = None,
    max_epochs = 1000,
    max_epochs_pre_ae = 100,
    max_epochs_pre_cl = 100,
    max_epochs_pre_sr = 100,
    max_epochs_pre_bd = 100,
    max_epochs_pre_bc = 100,
    max_epochs_pre_dr = 100,
    learning_rate = 1e-5, 
    adversarial_learning_rate = 1e-5, 
    pre_learning_rate_ae = 1e-3,
    pre_learning_rate_cl = 1e-3,
    pre_learning_rate_sr = 1e-3,
    pre_learning_rate_bd = 1e-3,
    pre_learning_rate_bc = 1e-3,
    pre_learning_rate_dr = 1e-3,
    early_stopping_set_denominator = 10, # = number of folds of which one is used
    early_stopping_set_stratify = True,
    model_args = {},
    data_batch_args = {}, 
    train_args = {},
    patient_datasize = None,
    cl_datasize = None, 
    pre_train_ae = True,
    pre_train_cl = True,
    pre_train_sr = True,
    pre_train_bd = True,
    pre_train_bc = True, 
    pre_train_dr = True,
    optimizer_class = tf.keras.optimizers.Adam, 
    optimizer_args = {'beta_1' : 0.9, 'beta_2' : 0.999}, 
    adversarial_optimizer_class = tf.keras.optimizers.experimental.SGD, 
    adversarial_optimizer_args = {}, 
    correction_optimizer_class = tf.keras.optimizers.experimental.SGD, 
    correction_optimizer_args = {}, 
    return_pre_trained_model_weights = False,
    debug = False
):
    patient_serialized = patient_serialized_dataset is not None
    cl_serialized = cl_serialized_dataset is not None
    train_args['patient_data'] = patient_serialized
    train_args['cell_line_data'] = cl_serialized
    
    if patient_serialized and cl_serialized:
        # Only one dataset can be longest, shorter ones are repeated
        patient_repeat = patient_datasize < cl_datasize
        cl_repeat = patient_datasize > cl_datasize
    else:
        patient_repeat, cl_repeat = (False, False)
    if patient_serialized:
        data_batch_args['shuffle'] = False
        data_batch_args['repeat'] = patient_repeat
        patient_sd = dataset_batch_setup(patient_serialized_dataset, **data_batch_args)
        if patient_serialized_test_dataset is not None:
            patient_test_sd = dataset_batch_setup(patient_serialized_test_dataset, **data_batch_args)
    if cl_serialized:
        data_batch_args['shuffle'] = False
        data_batch_args['repeat'] = cl_repeat
        cl_sd = dataset_batch_setup(cl_serialized_dataset, **data_batch_args)
        if cl_serialized_test_dataset is not None:
            cl_test_sd = dataset_batch_setup(cl_serialized_test_dataset, **data_batch_args)
    if patient_serialized or cl_serialized:
        train_args['tuple_dataset'] = False
        # Serialized datasets are already processed
        if patient_serialized and cl_serialized:
            train_args['train_dataset'] = patient_sd
            train_args['train_dataset2'] = cl_sd
            if patient_serialized_test_dataset is not None:
                train_args['valid_dataset'] = patient_test_sd
            if cl_serialized_test_dataset is not None:
                train_args['valid_dataset2'] = cl_test_sd
            model_args['batch_number'] = 2
            train_args['batch_patient_indicator'] = np.array([True, False])
        elif patient_serialized:
            train_args['train_dataset'] = patient_sd
            if patient_serialized_test_dataset is not None:
                train_args['valid_dataset'] = patient_test_sd
            train_args['batch_patient_indicator'] = np.array([True])
        else:
            train_args['train_dataset'] = cl_sd
            if cl_serialized_test_dataset is not None:
                train_args['valid_dataset'] = cl_test_sd
            train_args['batch_patient_indicator'] = np.array([False])
    else:
        if model_args.get('batch_adversarial_model', False):
            b_u = np.unique(b)
            b_n = b_u.shape[0]
            # Assume only one unique value can be < 0
            # TODO: add ValueErrors here and above?
            if np.any(b_u < 0):
                b_n -= 1
            model_args['batch_number'] = b_n
        
        if model_args.get('drug_response_model', False):
            model_args['drug_number'] = dr.shape[1]
        
        numpy_dataset_args = {}
        if train_args.get('early_stopping', False):
            if not model_args.get('supervised', False):
                early_stopping_set_stratify = False # no y to stratify by
                # TODO: stratify by survival event
            cv_ind = cross_validation_index(
                N = x.shape[0], 
                nfolds = early_stopping_set_denominator, 
                labs = y, 
                stratified = early_stopping_set_stratify)
            
            # Define hold-out set for cost monitoring
            numpy_dataset_args['x_train'] = x[cv_ind != 0,:]
            numpy_dataset_args['x_valid'] = x[cv_ind == 0,:]
            
            if model_args.get('supervised', False):
                numpy_dataset_args['y_train'] = y[cv_ind != 0]
                numpy_dataset_args['y_valid'] = y[cv_ind == 0]
            
            if model_args.get('survival_model', False):
                numpy_dataset_args['event_train'] = events[cv_ind != 0]
                numpy_dataset_args['event_valid'] = events[cv_ind == 0]
                numpy_dataset_args['time_train'] = times[cv_ind != 0]
                numpy_dataset_args['time_valid'] = times[cv_ind == 0]
                numpy_dataset_args['covariates_train'] = covariates[cv_ind != 0,:]
                numpy_dataset_args['covariates_valid'] = covariates[cv_ind == 0,:]
            
            if model_args.get('batch_adversarial_model', False):
                numpy_dataset_args['b_train'] = b[cv_ind != 0]
                numpy_dataset_args['b_valid'] = b[cv_ind == 0]
            
            if model_args.get('drug_response_model', False):
                numpy_dataset_args['dr_train'] = dr[cv_ind != 0,:]
                numpy_dataset_args['dr_valid'] = dr[cv_ind == 0,:]
        else:
            numpy_dataset_args['x_train'] = x
            numpy_dataset_args['y_train'] = y
            if model_args.get('survival_model', False):
                numpy_dataset_args['event_train'] = events
                numpy_dataset_args['time_train'] = times
                numpy_dataset_args['covariates_train'] = covariates
            if model_args.get('batch_adversarial_model', False):
                numpy_dataset_args['b_train'] = b
            if model_args.get('drug_response_model', False):
                numpy_dataset_args['dr_train'] = dr
        
        train_dataset, valid_dataset = prepare_tf_dataset_from_numpy(**numpy_dataset_args)
        train_args['train_dataset'] = dataset_batch_setup(train_dataset, **data_batch_args)
        if valid_dataset is not None:
            train_args['valid_dataset'] = dataset_batch_setup(valid_dataset, **data_batch_args)
    
    if debug:
        for k,i in model_args.items():
            print('Model args \'' + str(k) + '\' value: ' + str(i), file = sys.stderr)
    model = MODAE(**model_args)
    train_args['model'] = model
    train_args['verbose'] = False
    
    results = {'model' : model}
    if pre_train_ae:
        pre_train_ae_optimizer = optimizer_class(pre_learning_rate_ae, **optimizer_args)
        train_args_ae = train_args.copy()
        train_args_ae['optimizer'] = pre_train_ae_optimizer
        train_args_ae['epochs'] = max_epochs_pre_ae
        train_args_ae['train_ae_only'] = True
        
        results['train_losses_ae'], results['valid_losses_ae'] = train(**train_args_ae)
        if return_pre_trained_model_weights:
            results['model_ae_pre_weights'] = copy(model.get_weights())
    
    if model.batch_adversarial_model and pre_train_bd:
        pre_train_bd_optimizer = adversarial_optimizer_class(pre_learning_rate_bd, **adversarial_optimizer_args)
        train_args_bd = train_args.copy()
        train_args_bd['adversarial_optimizer'] = pre_train_bd_optimizer
        train_args_bd['epochs'] = max_epochs_pre_bd
        train_args_bd['train_batch_detector_only'] = True
        
        results['train_losses_bd'], results['valid_losses_bd'] = train(**train_args_bd)
        if return_pre_trained_model_weights:
            results['model_bd_pre_weights'] = copy(model.get_weights())
    
    if model.batch_adversarial_model and pre_train_bc:
        pre_train_bc_optimizer = correction_optimizer_class(pre_learning_rate_bc, **correction_optimizer_args)
        pre_train_bc_adversarial_optimizer = adversarial_optimizer_class(pre_learning_rate_bc, **adversarial_optimizer_args)
        train_args_bc = train_args.copy()
        train_args_bc['optimizer'] = pre_train_bc_optimizer
        train_args_bc['adversarial_optimizer'] = pre_train_bc_adversarial_optimizer
        train_args_bc['epochs'] = max_epochs_pre_bc
        train_args_bc['train_batch_correction_only'] = True
        
        results['train_losses_bc'], results['valid_losses_bc'] = train(**train_args_bc)
        if return_pre_trained_model_weights:
            results['model_bc_pre_weights'] = copy(model.get_weights())
    
    if model.supervised and pre_train_cl:
        pre_train_cl_optimizer = optimizer_class(pre_learning_rate_cl, **optimizer_args)
        train_args_cl = train_args.copy()
        train_args_cl['optimizer'] = pre_train_cl_optimizer
        train_args_cl['epochs'] = max_epochs_pre_cl
        train_args_cl['train_classifier_only'] = True
        
        results['train_losses_cl'], results['valid_losses_cl'] = train(**train_args_cl)
        if return_pre_trained_model_weights:
            results['model_cl_pre_weights'] = copy(model.get_weights())
    
    if model.survival_model and pre_train_sr:
        pre_train_sr_optimizer = optimizer_class(pre_learning_rate_sr, **optimizer_args)
        train_args_sr = train_args.copy()
        train_args_sr['optimizer'] = pre_train_sr_optimizer
        train_args_sr['epochs'] = max_epochs_pre_sr
        train_args_sr['train_survival_only'] = True
        
        results['train_losses_sr'], results['valid_losses_sr'] = train(**train_args_sr)
        if return_pre_trained_model_weights:
            results['model_sr_pre_weights'] = copy(model.get_weights())
    
    if model.drug_response_model and pre_train_dr:
        pre_train_dr_optimizer = optimizer_class(pre_learning_rate_dr, **optimizer_args)
        train_args_dr = train_args.copy()
        train_args_dr['optimizer'] = pre_train_dr_optimizer
        train_args_dr['epochs'] = max_epochs_pre_dr
        train_args_dr['train_drugs_only'] = True
        
        results['train_losses_dr'], results['valid_losses_dr'] = train(**train_args_dr)
        if return_pre_trained_model_weights:
            results['model_dr_pre_weights'] = copy(model.get_weights())
    
    optimizer = optimizer_class(learning_rate, **optimizer_args)
    train_args['optimizer'] = optimizer
    train_args['epochs'] = max_epochs
    if model.batch_adversarial_model:
        if pre_train_bc:
            adversarial_optimizer = pre_train_bc_adversarial_optimizer
        elif pre_train_bd:
            adversarial_optimizer = pre_train_bd_optimizer
        else:
            adversarial_optimizer = optimizer_class(adversarial_learning_rate, **optimizer_args)
        if pre_train_bc or pre_train_bd:
            adversarial_optimizer.lr.assign(adversarial_learning_rate)
            for opt_var in adversarial_optimizer.variables():
                opt_var.assign(tf.zeros_like(opt_var))
        train_args['adversarial_optimizer'] = adversarial_optimizer
    
    results['train_losses'], results['valid_losses'] = train(**train_args)
    
    return results

'''
Single step of CV random parameter search

Runs one setting generated from the given generator functions for one fold in. 
Repeated cross-validation. Meant to be run via parallel_parameter_search_iter_cv. 
'''
def parameter_search_iter(
    data_dict, 
    runs = [], # CV repeats to run in this thread
    folds = [], # CV folds to run in this thread
    nruns = 1, # CV repeats to generate from cv_seed
    nfolds = 5, # CV folds to generate from cv_seed 
    stratify_survival = True,
    stratify_subtype = True,
    stratify_time_quantiles = 10,
    cv_seed = 0, 
    model_args = {},
    data_batch_args = {},
    train_args = {},
    gym_args = {}, 
    data_standardize = False,
    task_id = None,
    save = False,
    file_name_prefix = '',
    omics_layer = 'mrna',
    parallel = False,
    gpu_memory = None,
    nthreads_interop = 2,
    nthreads = 4,
    model_metrics = True,
    pre_trained_model_metrics = True,
    survival_evaluation_brier_times = None,
    ps_validation_sets = True, 
    ps_test_sets = True,
):
    # Do cv splits within this function, for maximum compatibility.
    # Even if it is run in separate threads, fixing the seed makes it reproducible
    N = data_dict.get('x')[omics_layer].shape[0]
    stratify = False
    strat_var = np.full((N,),'')
    if model_args['supervised'] and stratify_subtype:
        stratify = True
        strat_var = np.char.add(strat_var, data_dict.get('y').astype('str'))
    if model_args['survival_model'] and stratify_survival:
        stratify = True
        sevent = data_dict.get('survival_event')
        stimes = data_dict.get('survival_time')
        stq = np.quantile(stimes[stimes >= 0], np.arange(1,stratify_time_quantiles) / stratify_time_quantiles)
        stbins = np.digitize(stimes, stq)
        stbins[stimes < 0] = -1 # Set all ambiguous/NaNs to -1 (own bin)
        strat_var = np.char.add(strat_var, sevent.astype('str'))
        strat_var = np.char.add(strat_var, stbins.astype('str'))
    
    if not save:
        raise Exception('Return not implemented, must save to file.')
    
    '''If running concurrently with other processes we need to limit GPU memory usage or cpu usage'''
    if parallel:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) and gpu_memory is not None:
            print('Found {} GPUs, using 1 logical GPU with {} MB memory.'.format(len(gpus), gpu_memory))
            '''Fixed allocation, less overhead and easy to calculate'''
            tf.config.set_logical_device_configuration(
                    gpus[0], # assume only one gpu
                    [tf.config.LogicalDeviceConfiguration(memory_limit = gpu_memory)]
            )
            logical_gpus = tf.config.list_logical_devices('GPU')
        else:
            tf.config.threading.set_inter_op_parallelism_threads(nthreads_interop)
            tf.config.threading.set_intra_op_parallelism_threads(nthreads)
    
    if len(folds) == 0:
        folds = np.arange(nfolds)
    if len(runs) == 0:
        runs = np.arange(nfolds)
    for run in runs:
        cv_split = cross_validation_index(N = N, 
                                          nfolds = nfolds,
                                          random_seed = cv_seed + run,
                                          labs = strat_var, 
                                          stratified = stratify)
        # Use fixed seed for parameter initialization for each run
        np.random.seed(run)
        model_args['encoder_init_seeds'] = np.random.randint(2**31, size = len(model_args['encoder_layers']), dtype=np.int32)
        model_args['decoder_init_seeds'] = np.random.randint(2**31, size = len(model_args['decoder_layers']), dtype=np.int32)
        model_args['recon_init_seed'] = np.random.randint(2**31, size = 1, dtype=np.int32)[0]
        model_args['classifier_init_seeds'] = np.random.randint(2**31, size = len(model_args['classifier_layers']), dtype=np.int32)
        model_args['classifier_final_init_seed'] = np.random.randint(2**31, size = 1, dtype=np.int32)[0]
        model_args['survival_model_init_seeds'] = np.random.randint(2**31, size = len(model_args['survival_model_layers']), dtype=np.int32)
        model_args['survival_model_final_init_seed'] = np.random.randint(2**31, size = 1, dtype=np.int32)[0]
        
        for fold in folds:
            # Add validation and test indices for training
            train_inds = []
            test_inds = []
            set_file_prefix = []
            if ps_validation_sets:
                # Evaluate parameters by training on k-2 folds and evaluating on 1 fold
                train_inds.append(np.logical_and(cv_split != fold, cv_split != ((fold + 1) % nfolds)))
                test_inds.append(cv_split == ((fold + 1) % nfolds))
                set_file_prefix.append('ps_cv_')
            if ps_test_sets:
                # Evaluate parameters by training on k-1 folds and evaluating on 1 fold
                train_inds.append(cv_split != fold)
                test_inds.append(cv_split == fold)
                set_file_prefix.append('test_cv_')
            
            for cv_train_ind, cv_test_ind, file_string in zip(train_inds, test_inds, set_file_prefix):
                processed_data, model_spec = process_patient_data(
                    x = data_dict.get('x'), 
                    cv_train_ind = cv_train_ind, 
                    cv_test_ind = cv_test_ind, 
                    supervised = model_args['supervised'],
                    survival_model = model_args['survival_model'],
                    y = data_dict.get('y', None), 
                    survival_event = data_dict.get('survival_event', None),
                    survival_time = data_dict.get('survival_time', None), 
                    survival_covariates = data_dict.get('survival_covariates', None),
                    survival_covariates_categorical = data_dict.get('survival_covariates_categorical', None),
                    data_standardize = data_standardize)
                
                processed_data['x_train'] = processed_data['x_train'][omics_layer]
                processed_data['x_test'] = processed_data['x_test'][omics_layer]
                model_spec['input_dim'] = model_spec['input_dim'][omics_layer]
                processed_data['rownames_train'] = data_dict.get('ind').index[cv_train_ind]
                processed_data['rownames_test'] = data_dict.get('ind').index[cv_test_ind]
                
                result = train_aecl_with_pretraining(
                    x = processed_data['x_train'],
                    y = processed_data['y_train'],
                    events = processed_data['survival_event_train'],
                    times = processed_data['survival_time_train'],
                    covariates = processed_data['survival_covariates_train'],
                    model_args = {**model_args, **model_spec},
                    data_batch_args = data_batch_args,
                    train_args = train_args,
                    **gym_args, 
                    return_pre_trained_model_weights = True)
                evaluate_results(
                    result, 
                    data = processed_data,
                    run = run, 
                    fold = fold,
                    task_id = task_id,
                    save = save,
                    file_name_prefix = file_name_prefix + file_string,
                    model_metrics = model_metrics, 
                    pre_trained_model_metrics = pre_trained_model_metrics, 
                    survival_evaluation_brier_times = survival_evaluation_brier_times)
    return 0

def serialized_parameter_search_iter(
    serialized_data, 
    runs = [], # CV repeats to run in this thread
    folds = [], # CV folds to run in this thread
    nruns = 1, # CV repeats to generate from cv_seed
    nfolds = 5, # CV folds to generate from cv_seed 
    stratify_survival = True,
    stratify_subtype = True,
    stratify_time_quantiles = 10,
    cv_seed = 0, 
    model_args = {},
    data_batch_args = {}, 
    train_args = {},
    gym_args = {}, 
    data_standardize = False,
    task_id = None,
    save = False,
    file_name_prefix = '',
    omics_layer = 'mrna',
    parallel = False,
    gpu_memory = None,
    nthreads_interop = 2,
    nthreads = 4,
    model_metrics = True,
    pre_trained_model_metrics = True,
    survival_evaluation_brier_times = None,
    survival_evaluation_brier = False
):
    if not save:
        raise Exception('Return not implemented, must save to file.')
    
    '''If running concurrently with other processes we need to limit GPU memory usage or cpu usage'''
    if parallel:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) and gpu_memory is not None:
            print('Found {} GPUs, using 1 logical GPU with {} MB memory.'.format(len(gpus), gpu_memory))
            '''Fixed allocation, less overhead and easy to calculate'''
            tf.config.set_logical_device_configuration(
                    gpus[0], # assume only one gpu
                    [tf.config.LogicalDeviceConfiguration(memory_limit = gpu_memory)]
            )
            logical_gpus = tf.config.list_logical_devices('GPU')
        else:
            tf.config.threading.set_inter_op_parallelism_threads(nthreads_interop)
            tf.config.threading.set_intra_op_parallelism_threads(nthreads)
    
    serialized_data = JSONFeatureSpecDecoder(serialized_data)
    
    if len(folds) == 0:
        folds = np.arange(nfolds)
    if len(runs) == 0:
        runs = np.arange(nfolds)
    for run in runs:
        # Use fixed seed for parameter initialization for each run
        np.random.seed(run)
        if model_args.get('encoder_layers', None):
            model_args['encoder_init_seeds'] = np.random.randint(2**31, size = len(model_args.get('encoder_layers')), dtype=np.int32)
        if model_args.get('decoder_layers', None):
            model_args['decoder_init_seeds'] = np.random.randint(2**31, size = len(model_args.get('decoder_layers')), dtype=np.int32)
        model_args['recon_init_seed'] = np.random.randint(2**31, size = 1, dtype=np.int32)[0]
        if model_args.get('classifier_layers', None):
            model_args['classifier_init_seeds'] = np.random.randint(2**31, size = len(model_args.get('classifier_layers')), dtype=np.int32)
        model_args['classifier_final_init_seed'] = np.random.randint(2**31, size = 1, dtype=np.int32)[0]
        if model_args.get('survival_model_layers', None):
            model_args['survival_model_init_seeds'] = np.random.randint(2**31, size = len(model_args.get('survival_model_layers')), dtype=np.int32)
        model_args['survival_model_final_init_seed'] = np.random.randint(2**31, size = 1, dtype=np.int32)[0]
        if model_args.get('batch_adversarial_model_layers', None):
            model_args['batch_adversarial_model_init_seeds'] = np.random.randint(2**31, size = len(model_args.get('batch_adversarial_model_layers')), dtype=np.int32)
        model_args['batch_adversarial_model_pred_seed'] = np.random.randint(2**31, size = 1, dtype=np.int32)[0]
        if model_args.get('drug_response_model_layers', None):
            model_args['drug_response_model_init_seeds'] = np.random.randint(2**31, size = len(model_args.get('drug_response_model_layers')), dtype=np.int32)
        if model_args.get('drug_response_model_drugwise_layers', None):
            model_args['drug_response_model_drugwise_init_seeds'] = np.random.randint(2**31, size = len(model_args.get('drug_response_model_drugwise_layers')), dtype=np.int32)
        model_args['drug_response_model_pred_seed'] = np.random.randint(2**31, size = 1, dtype=np.int32)[0]
        
        for fold in folds:
            data_instance = serialized_data[str(run)][str(fold)]
            for file_string in data_instance.keys():
                # Load datasets
                datasets = list(data_instance[file_string].keys())
                if np.isin('patient', datasets):
                    patient_training_dataset = parse_serialized_dataset_from_file(
                        serialized_file = data_instance[file_string]['patient']['train']['filename'],
                        feature_spec = data_instance[file_string]['patient']['train']['feature_spec'])
                    patient_testing_dataset = parse_serialized_dataset_from_file(
                        serialized_file = data_instance[file_string]['patient']['test']['filename'],
                        feature_spec = data_instance[file_string]['patient']['test']['feature_spec'])
                    patient_training_rows = np.array(data_instance[file_string]['patient']['train']['sample_info']['rownames'])
                    patient_testing_rows = np.array(data_instance[file_string]['patient']['test']['sample_info']['rownames'])
                    patient_model_spec = data_instance[file_string]['patient']['train']['model_spec']
                    patient_datasize = len(data_instance[file_string]['patient']['train']['sample_info']['rownames'])
                else:
                    patient_training_dataset = None
                    patient_testing_dataset = None
                    patient_training_rows = None
                    patient_testing_rows = None
                    patient_model_spec = {}
                if np.isin('cell_line', datasets):
                    cl_training_dataset = parse_serialized_dataset_from_file(
                        serialized_file = data_instance[file_string]['cell_line']['train']['filename'],
                        feature_spec = data_instance[file_string]['cell_line']['train']['feature_spec'])
                    cl_testing_dataset = parse_serialized_dataset_from_file(
                        serialized_file = data_instance[file_string]['cell_line']['test']['filename'],
                        feature_spec = data_instance[file_string]['cell_line']['test']['feature_spec'])
                    cl_training_rows = np.array(data_instance[file_string]['cell_line']['train']['sample_info']['rownames'])
                    cl_testing_rows = np.array(data_instance[file_string]['cell_line']['test']['sample_info']['rownames'])
                    cl_model_spec = data_instance[file_string]['cell_line']['train']['model_spec']
                    cl_datasize = len(data_instance[file_string]['cell_line']['train']['sample_info']['rownames'])
                else:
                    cl_training_dataset = None
                    cl_testing_dataset = None
                    cl_training_rows = None
                    cl_testing_rows = None
                    cl_model_spec = {}
                if np.isin('patient', datasets) and np.isin('cell_line', datasets):
                    cl_in = cl_model_spec.pop('input_dim')
                    if cl_in != patient_model_spec['input_dim']:
                        raise ValueError('Input dimensions for serialized patient and cell-line data do not match.')
                data_model_spec = {**patient_model_spec, **cl_model_spec}
                # Run parameter search
                result = train_aecl_with_pretraining(
                    patient_serialized_dataset = patient_training_dataset, 
                    patient_serialized_test_dataset = patient_testing_dataset, 
                    cl_serialized_dataset = cl_training_dataset, 
                    cl_serialized_test_dataset = cl_testing_dataset, 
                    model_args = {**model_args, **data_model_spec},
                    data_batch_args = data_batch_args, 
                    train_args = train_args, 
                    patient_datasize = patient_datasize,
                    cl_datasize = cl_datasize, 
                    **gym_args, 
                    return_pre_trained_model_weights = True)
                data_batch_args['repeat'] = False
                evaluate_results(
                    result, 
                    patient_serialized_training_dataset = {
                        'dataset' : dataset_batch_setup(patient_training_dataset, **data_batch_args), 
                        'rows' : patient_training_rows, 
                        'name' : 'patient_train'},
                    patient_serialized_testing_dataset = {
                        'dataset' : dataset_batch_setup(patient_testing_dataset, **data_batch_args), 
                        'rows' : patient_testing_rows, 
                        'name' : 'patient_test'},
                    cl_serialized_training_dataset = {
                        'dataset' : dataset_batch_setup(cl_training_dataset, **data_batch_args), 
                        'rows' : cl_training_rows, 
                        'name' : 'cl_train'},
                    cl_serialized_testing_dataset = {
                        'dataset' : dataset_batch_setup(cl_testing_dataset, **data_batch_args), 
                        'rows' : cl_testing_rows, 
                        'name' : 'cl_test'},
                    run = run, 
                    fold = fold,
                    task_id = task_id,
                    save = save,
                    file_name_prefix = file_name_prefix + file_string,
                    model_metrics = model_metrics, 
                    pre_trained_model_metrics = pre_trained_model_metrics, 
                    survival_evaluation_brier_times = survival_evaluation_brier_times, 
                    survival_evaluation_brier = survival_evaluation_brier)
    return 0

def parallel_parameter_search_iter(
        max_cv_repeats = np.inf,
        process_timeout = np.inf,
        nprocess = 1,
        sleep_time = 5,
        iter_args = {},
        return_ps_metrics = False,
        serialize_data = False,
        ):
    nruns = iter_args['nruns']
    n_splits = nruns * iter_args['nfolds']
    
    '''
    Loop process spawner until done
    '''
    processes = []
    start_times = []
    split_i = 0
    finished_splits = 0
    while finished_splits < n_splits:
        '''Spawn until all runs spawned while limiting number of concurrent splits'''
        if split_i < n_splits and len(processes) < nprocess:
            iter_args_i = iter_args.copy()
            run_i = split_i % nruns
            fold_i = split_i // nruns
            
            iter_args_i['runs'] = [run_i]
            iter_args_i['folds'] = [fold_i]
            if serialize_data:
                _ = iter_args_i.pop('data_dict', None)
                _ = iter_args_i.pop('ps_validation_sets', None)
                _ = iter_args_i.pop('ps_test_sets', None)
                processes.append(Process(target = serialized_parameter_search_iter, kwargs = iter_args_i))
            else:
                processes.append(Process(target = parameter_search_iter, kwargs = iter_args_i))
            processes[-1].start()
            start_times.append(time.time())
            split_i += 1
        else:
            closed_i = []
            # Check all processes
            for i,p in enumerate(processes):
                # If process is running
                if p.is_alive():
                    # If process is overtime (never by default)
                    if time.time() - start_times[i] > process_timeout:
                        print('Process timeout.')
                        p.kill()
                        finished_splits += 1 # Split did not finish in time, do not retry
                else: # If not running
                    p.close()
                    closed_i.append(i)
                    finished_splits += 1 # Split finished
            # Remove all closed processes
            if len(closed_i) > 0:
                processes = [p for i,p in enumerate(processes) if not i in closed_i]
                start_times = [t for i,t in enumerate(start_times) if not i in closed_i]
            else:
                # Sleep before looping again
                time.sleep(sleep_time)
    
    '''Cleanup all the files created'''
    compression_map = {'.csv' : None, '.csv.gz' : 'gzip'}
    for string_i, index_i, format_i in [('embeddings', True, '.csv.gz'), 
                                        ('predictions', True, '.csv.gz'),
                                        ('metrics', False, '.csv'),
                                        ('diagnostics', False, '.csv.gz')]:
        # Deal with non-strictly typed parameters in pandas
        if index_i:
            index_ii = 0
        else:
            index_ii = False
        
        subtask_names = []
        if iter_args['ps_validation_sets']:
            subtask_names.append('ps_cv_')
        if iter_args['ps_test_sets']:
            subtask_names.append('test_cv_')
        for subtask_j in subtask_names:
            file_fragments = glob.glob(f"{iter_args['file_name_prefix']}{subtask_j}{string_i}_run*task{iter_args['task_id']}{format_i}")
            fragments = [pd.read_csv(i, header = 0, index_col = index_ii) for i in file_fragments]
            if len(fragments) > 0:
                combined = pd.concat(fragments)
                for file in file_fragments:
                    os.remove(file)
                combined.to_csv(f"{iter_args['file_name_prefix']}{subtask_j}{string_i}_task{iter_args['task_id']}{format_i}",
                                na_rep = 'NA', header = True, index = index_i, compression = compression_map[format_i])
    if return_ps_metrics:
        fname = iter_args['file_name_prefix'] + 'ps_cv_metrics_task{}'.format(iter_args['task_id']) + '.csv'
        if os.path.exists(fname):
            result = pd.read_csv(fname, header = 0, index_col = False)
        else:
            result = None
        return result.loc[result['stage'] == 'final']
