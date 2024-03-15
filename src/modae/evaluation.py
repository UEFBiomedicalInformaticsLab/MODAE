# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:51:15 2023

@author: rintala
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from copy import copy
import re
import sys

import sklearn.metrics as skl_metrics
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from scipy.special import softmax
from modae.survival import breslow_baseline_survival_time_estimate
from modae.data_batch import batch_to_step_args

def DSC(data_matrix, batch_label):
  u, idx, n = tf.unique_with_counts(batch_label)
  
  # Divide into blocks
  idx_sorted = tf.argsort(idx, direction = 'ASCENDING')
  
  bblox = tf.split(tf.gather(data_matrix, idx_sorted, axis = 0), n, axis = 0)
  
  p = n / tf.reduce_sum(n)
  
  Sw = []
  mu = []
  for i in range(len(bblox)):
      Sw.append(tf.math.reduce_sum(tf.math.reduce_variance(bblox[i], axis = 0)))
      mu.append(tf.math.reduce_mean(bblox[i], axis = 0))
  Dw = tf.math.sqrt(tf.math.reduce_sum(p * Sw))
  M = tf.math.reduce_sum(tf.expand_dims(p, axis = -1) * mu, axis = 0)
  Sb = tf.math.reduce_sum((mu - M)**2, axis = 1)
  Db = tf.math.sqrt(tf.math.reduce_sum(Sb * p))
  return Db / Dw

def get_embeddings(model, dataset, tuple_dataset = True):
    z_full_list = []
    for batch in tf.data.Dataset.zip((dataset, )):
        step_args = batch_to_step_args(
            batch, 
            tuple_dataset = tuple_dataset)
        if model.variational:
            z, logvar = model.encode(step_args['x'])
        else:
            z = model.encode(step_args['x'])
        z_full_list.append(z)
    return tf.concat(z_full_list, axis = 0)

def get_model_predictions(model, dataset, tuple_dataset = True):
    survival_risk_full = []
    survival_mask_full = []
    survival_hazard_full = []
    class_pred_full = []
    dr_pred_full = []
    batch_score_full = []
    for batch in tf.data.Dataset.zip((dataset, )):
        step_args = batch_to_step_args(
            batch, 
            tuple_dataset = tuple_dataset)
        if model.variational:
            z, logvar = model.encode(step_args['x'])
        else:
            z = model.encode(step_args['x'])
        if model.deconfounder_model:
            shared_embedding_mask = tf.math.reduce_all(
                model.batch_specific_layer_mask > 0, axis = 0)
            z = tf.boolean_mask(z, shared_embedding_mask, axis = 1)
        if model.supervised:
            class_pred = model.classifier_net(z)
            class_pred_full.append(class_pred)
        if model.drug_response_model:
            dr_pred = model.drug_response_net(z)
            if model.drugwise_model:
                dr_pred_dw = []
                for drug_i in range(model.drug_number):
                    drug_model_i = model.drug_response_drugwise_nets[drug_i]
                    dr_predi = drug_model_i(dr_pred)
                    dr_pred_dw.append(dr_predi)
                dr_pred = tf.concat(dr_pred_dw, axis = 1)
            dr_pred_full.append(dr_pred)
        if model.batch_adversarial_model:
            batch_score = model.batch_detector(z)
            batch_score_full.append(batch_score)
        if model.survival_covariate_n > 0:
            # Cannot predict without covariates
            surv_step_args = batch_to_step_args(
                batch, 
                survival = True, 
                tuple_dataset = tuple_dataset)
            if surv_step_args.get('x', None) is not None:
                if model.variational:
                    z, logvar = model.encode(surv_step_args['x'])
                else:
                    z = model.encode(surv_step_args['x'])
                if model.deconfounder_model:
                    shared_embedding_mask = tf.math.reduce_all(
                        model.batch_specific_layer_mask > 0, axis = 0)
                    z = tf.boolean_mask(z, shared_embedding_mask, axis = 1)
                z_a = tf.concat((z, surv_step_args['survival_covariates']), axis = 1)
                z_a = tf.boolean_mask(z_a, mask = surv_step_args['survival_mask'])
                hazard_components = model.hazard_component_net(z_a)
                survival_hazard_full.append(hazard_components)
                survival_risk = model.survival_risk_net(hazard_components)
                survival_risk_full.append(survival_risk)
                survival_mask_full.append(surv_step_args['survival_mask'])
    out = {}
    if len(survival_risk_full) > 0:
        survival_risk = tf.concat(survival_risk_full, axis = 0)
        survival_mask = tf.concat(survival_mask_full, axis = 0)
        out['survival_risk'] = np.full((survival_mask.shape[0], 1), np.nan)
        out['survival_risk'][survival_mask.numpy()] = survival_risk.numpy()
        survival_hazard = tf.concat(survival_hazard_full, axis = 0)
        out['survival_hazard'] = np.full(
            (survival_mask.shape[0], survival_hazard.shape[1]), 
            np.nan)
        out['survival_hazard'][survival_mask.numpy()] = survival_hazard.numpy()
    if len(class_pred_full) > 0:
        out['class_pred'] = tf.concat(class_pred_full, axis = 0)
    if len(dr_pred_full) > 0:
        out['dr_pred'] = tf.concat(dr_pred_full, axis = 0)
    if len(batch_score_full) > 0:
        out['batch_score'] = tf.concat(batch_score_full, axis = 0)
    return out

def get_model_losses(
        model, 
        dataset, 
        dataset2 = None, 
        tuple_dataset = True, 
        patient_and_cl = False, # not used, but could help
        get_metrics = False,
        brier = False, 
        brier_times = None, 
        debug = False, 
        manual_batch_flag = False):
    if debug:
        print('Evaluation model layer shapes:' , file = sys.stderr)
        for w in model.get_weights():
            print(w.shape, file = sys.stderr)
    # Define collectors over mini-batches
    loss_reconstruction_ds1 = tf.keras.metrics.Mean()
    loss_reconstruction_ds2 = tf.keras.metrics.Mean()
    loss_deconfounding = tf.keras.metrics.Mean()
    loss_supervised = tf.keras.metrics.Mean()
    loss_survival = tf.keras.metrics.Mean()
    loss_batch = tf.keras.metrics.Mean()
    loss_batch_dsc = tf.keras.metrics.Mean()
    loss_drugs = tf.keras.metrics.Mean()
    if get_metrics:
        '''
        recon_r2 = tf.keras.metrics.R2Score(class_aggregation = 'uniform_average')
        '''
        if model.supervised:
            acc = tf.keras.metrics.CategoricalAccuracy()
            auroc = tf.keras.metrics.AUC(curve = 'ROC', from_logits = True)
            aupr = tf.keras.metrics.AUC(curve = 'PR', from_logits = True)
        if model.survival_model:
            risk_vecs = []
            time_vecs = []
            event_vecs = []
        '''
        if model.drug_response_model:
            drug_r2 = [tf.keras.metrics.R2Score(class_aggregation = 'uniform_average') \
                       for i in range(model.drug_number)]
        '''
    if dataset2 is not None:
        iterator = tf.data.Dataset.zip((dataset, dataset2))
    else:
        iterator = tf.data.Dataset.zip((dataset, ))
    for batch in iterator:
        step_args1 = batch_to_step_args(
            batch, 
            batches = True, 
            tuple_dataset = tuple_dataset)
        if debug:
            step_arg_tensor_shapes = (
                dict([(k, i.shape) for k,i in step_args1.items() if isinstance(i, tf.Tensor)]))
            for k,i in step_arg_tensor_shapes.items():
                print('AE and batch evaluation step args \'' + str(k) + 
                      '\' shape: ' + str(i), file = sys.stderr)
        recon_loss, z_full = model.compute_loss_unsupervised(
            x = step_args1['x'], b = step_args1['b'])
        '''
        if get_metrics:
            recon_x, recon_std = model.decode(z_full, b = step_args1['b'])
            recon_r2.update_state(step_args1['x'], recon_x)
        '''
        if model.deconfounder_model:
            loss_deconfounding.update_state(model.confounder_loss(
                z_full, b = step_args1['b']), sample_weight = step_args1['x'].shape[0])
        # TODO: re-implement to add support for more batches
        loss_reconstruction_ds1.update_state(
            tf.boolean_mask(recon_loss, step_args1['b'] == 0, axis = 0))
        loss_reconstruction_ds2.update_state(
            tf.boolean_mask(recon_loss, step_args1['b'] == 1, axis = 0))
        if model.supervised:
            step_args = batch_to_step_args(
                batch, 
                tuple_dataset = tuple_dataset,
                classes = True)
            if debug:
                step_arg_tensor_shapes = (
                    dict([(k, i.shape) for k,i in step_args.items() if isinstance(i, tf.Tensor)]))
                for k,i in step_arg_tensor_shapes.items():
                    print('Classifier evaluation step args \'' + str(k) + 
                          '\' shape: ' + str(i), file = sys.stderr)
            if step_args.get('x', None) is not None:
                if model.variational:
                    z, logvar = model.encode(step_args['x'])
                else:
                    z = model.encode(step_args['x'])
                if model.deconfounder_model:
                    shared_embedding_mask = tf.math.reduce_all(
                        model.batch_specific_layer_mask > 0, axis = 0)
                    z = tf.boolean_mask(z, shared_embedding_mask, axis = 1)
                supervised_loss, y_logit, n_labeled = model.compute_loss_supervised(
                    z = z, 
                    y = step_args.get('y', None), 
                    mask = step_args.get('y_mask', None))
                loss_supervised.update_state(supervised_loss)
                if get_metrics:
                    compute_supervised_metrics(
                        model, 
                        y_logit = y_logit, 
                        y = step_args.get('y', None), 
                        mask = step_args.get('y_mask', None), 
                        acc = acc, 
                        auroc = auroc, 
                        aupr = aupr)
        if model.survival_model:
            step_args = batch_to_step_args(
                batch, 
                tuple_dataset = tuple_dataset,
                survival = True)
            if debug:
                step_arg_tensor_shapes = (
                    dict([(k, i.shape) for k,i in step_args.items() if isinstance(i, tf.Tensor)]))
                for k,i in step_arg_tensor_shapes.items():
                    print('Survival evaluation step args \'' + str(k) + 
                          '\' shape: ' + str(i), file = sys.stderr)
            if step_args.get('x', None) is not None:
                if model.variational:
                    z, logvar = model.encode(step_args['x'])
                else:
                    z = model.encode(step_args['x'])
                if model.deconfounder_model:
                    shared_embedding_mask = tf.math.reduce_all(
                        model.batch_specific_layer_mask > 0, axis = 0)
                    z = tf.boolean_mask(z, shared_embedding_mask, axis = 1)
                survival_loglikelihood, n_events = model.compute_survival_loss_coxph(
                    z = z, 
                    times = step_args.get('survival_times'), 
                    events = step_args.get('survival_events'), 
                    mask = step_args.get('survival_mask'), 
                    covariates = step_args.get('survival_covariates', None))
                loss_survival.update_state(survival_loglikelihood)
                if get_metrics:
                    survival_vars = get_survival_eval_vars(
                        model = model, 
                        z = z, 
                        time = step_args.get('survival_times'), 
                        event = step_args.get('survival_events'), 
                        mask = step_args.get('survival_mask'), 
                        covariates = step_args.get('survival_covariates', None))
                    risk_vecs.append(survival_vars['risk'])
                    time_vecs.append(survival_vars['time'])
                    event_vecs.append(survival_vars['event'])
        if model.batch_adversarial_model:
            if manual_batch_flag or (dataset2 is not None):
                batch_flag = True
            else:
                u, idx = tf.unique(step_args1['b'])
                batch_flag = u.shape[0] > 1
            if batch_flag:
                if model.deconfounder_model:
                    shared_embedding_mask = tf.math.reduce_all(
                        model.batch_specific_layer_mask > 0, axis = 0)
                    z_full = tf.boolean_mask(z_full, shared_embedding_mask, axis = 1)
                batch_ce = model.batch_correction_loss(z_full, step_args1['b'])
                batch_dsc = DSC(z_full, step_args1['b'])
                loss_batch.update_state(batch_ce)
                loss_batch_dsc.update_state(batch_dsc)
        if model.drug_response_model:
            step_args = batch_to_step_args(
                batch, 
                tuple_dataset = tuple_dataset,
                drugs = True)
            if debug:
                step_arg_tensor_shapes = (
                    dict([(k, i.shape) for k,i in step_args.items() if isinstance(i, tf.Tensor)]))
                for k, i in step_arg_tensor_shapes.items():
                    print('Drug response evaluation step args \'' + str(k) + 
                          '\' shape: ' + str(i), file = sys.stderr)
            if step_args.get('x', None) is not None:
                if model.variational:
                    z, logvar = model.encode(step_args['x'])
                else:
                    z = model.encode(step_args['x'])
                if model.deconfounder_model:
                    shared_embedding_mask = tf.math.reduce_all(
                        model.batch_specific_layer_mask > 0, axis = 0)
                    z = tf.boolean_mask(z, shared_embedding_mask, axis = 1)
                dr_sse, n_screened = model.drug_loss(
                    z = z, 
                    dr = step_args.get('dr'), 
                    mask = step_args.get('dr_mask'))
                loss_drugs.update_state(dr_sse / n_screened, sample_weight = n_screened)
                '''
                if get_metrics:
                    dr_pred = model.drug_response_net(z)
                    dr_pred_masked = tf.ragged.boolean_mask(
                        tf.transpose(dr_pred), 
                        tf.transpose(step_args.get('dr_mask')))
                    dr_masked = tf.ragged.boolean_mask(
                        tf.transpose(step_args.get('dr')), 
                        tf.transpose(step_args.get('dr_mask')))
                    for i in range(len(drug_r2)):
                        drug_r2[i].update_state(
                            tf.transpose(dr_masked[i, None]), 
                            tf.transpose(dr_pred_masked[i, None]))
                '''
    
    if dataset2 is not None:
        out = {'reconstruction_loss_dataset1' : loss_reconstruction_ds1.result().numpy(), 
               'reconstruction_loss_dataset2' : loss_reconstruction_ds2.result().numpy()}
    else:
        total_mse = loss_reconstruction_ds1.result() + loss_reconstruction_ds2.result()
        out = {'reconstruction_loss_dataset1' : total_mse.numpy(), 
               'reconstruction_loss_dataset2' : 0.}
        '''
        if get_metrics:
            out['reconstruction_r2_score'] = recon_r2.result().numpy()
        '''
    out['regularization'] = tf.math.add_n(model.losses).numpy()
    if model.supervised:
        out['cross_entropy'] = loss_supervised.result().numpy()
        if get_metrics:
            out['auroc'] = auroc.result().numpy()
            out['aupr'] = aupr.result().numpy()
            out['acc'] = acc.result().numpy()
    if model.survival_model:
        out['survival_log_likelihood'] = loss_survival.result().numpy()
        if get_metrics:
            if len(risk_vecs) > 0:
                surv_metrics = get_survival_metrics(
                    risk = tf.concat(risk_vecs, axis = 0).numpy(), 
                    time = tf.concat(time_vecs, axis = 0).numpy(), 
                    event = tf.concat(event_vecs, axis = 0).numpy(), 
                    brier_times = brier_times, 
                    brier = brier)
                out = {**out, **surv_metrics}
    if model.batch_adversarial_model:
        out['batch_cross_entropy'] = loss_batch.result().numpy()
        out['batch_dsc'] = loss_batch_dsc.result().numpy()
    if model.drug_response_model:
        out['drug_response_mse'] = loss_drugs.result().numpy()
    if model.deconfounder_model:
        out['confounder_alignment_norm'] = loss_deconfounding.result().numpy()
    return out

def print_diagnostics(
    model, 
    metrics, 
    epoch, 
    start_time, 
    end_time, 
    supervised_metric, 
    set_name = 'validation set'
):
    if model.variational:
        loss_string = 'NELBO'
    else:
        loss_string = 'MSE'
    
    # Print validation set metrics
    monitor_string = ('Epoch: {}, ' + set_name + 
                      ' dataset1 {}: {}, dataset2 {}: {}, W: {}').format(
        epoch,
        loss_string,
        metrics['reconstruction_loss_dataset1'],
        loss_string,
        metrics['reconstruction_loss_dataset2'],
        metrics['regularization'])
    if model.supervised:
        monitor_string += (', predictions {}: {}').format(
            supervised_metric, 
            metrics[supervised_metric])
    if model.survival_model:
        monitor_string += (', survival log-likelihood: {}').format(
            metrics['survival_log_likelihood'])
    if model.batch_adversarial_model:
        monitor_string += (', batch DSC: {}').format(metrics['batch_dsc'])
        monitor_string += (', batch CE: {}').format(metrics['batch_cross_entropy'])
    if model.drug_response_model:
        monitor_string += (', drug-response MSE: {}').format(metrics['drug_response_mse'])
    monitor_string += (', time elapsed for current epoch {}').format(end_time - start_time)
    print(monitor_string)

def evaluate_results(
    result, 
    run, 
    fold, 
    task_id, 
    data = None, 
    patient_serialized_training_dataset = None, 
    patient_serialized_testing_dataset = None, 
    cl_serialized_training_dataset = None, 
    cl_serialized_testing_dataset = None, 
    save = True,
    file_name_prefix = '',
    model_metrics = True,
    pre_trained_model_metrics = True,
    pre_trained_model_embeddings = False,
    survival_evaluation_brier_times = None,
    survival_evaluation_brier = False
):
    model = result['model']
    
    datasets = [patient_serialized_training_dataset, 
                patient_serialized_testing_dataset, 
                cl_serialized_training_dataset, 
                cl_serialized_testing_dataset]
    dataset_rows = [i['rows'] for i in datasets if i is not None]
    dataset_names = [i['name'] for i in datasets if i is not None]
    datasets = [i['dataset'] for i in datasets if i is not None]
    tuple_dataset = len(datasets) == 0
    if tuple_dataset:
        # TODO: unpack tuple dataset from the data object
        #datasets = data
        #data
        for cv_set in []:
            dataset_rows = data['rownames_']
    
    weight_key_list = ['model_ae_pre_weights', 'model_cl_pre_weights', 
                       'model_sr_pre_weights', 'model_bd_pre_weights', 
                       'model_bc_pre_weights', 'model_dr_pre_weights']
    trained_list = [True]
    weight_list = [copy(model.get_weights())]
    weight_names = ['final']
    
    if pre_trained_model_metrics:
        trained_list += [result.get(i, None) is not None for i in weight_key_list]
        weight_list += [result.get(i, None) for i in weight_key_list]
        weight_names += ['ae_pt', 'cl_pt', 'sr_pt', 'bd_pt', 'bc_pt', 'dr_pt']
    
    dataset_pairs = [(patient_serialized_training_dataset, cl_serialized_training_dataset), 
                     (patient_serialized_testing_dataset, cl_serialized_testing_dataset)]
    dataset_pair_names = ['combined_train', 'combined_test']
    dataset_pair_names = [
        i for i, p in zip(dataset_pair_names, dataset_pairs) 
        if (p[0] is not None) and (p[1] is not None)]
    dataset_pairs = [
        (p[0]['dataset'], p[1]['dataset']) for p in dataset_pairs 
        if (p[0] is not None) and (p[1] is not None)]
    
    metrics_lists = {}
    for trained, weights, weight_name in zip(trained_list, weight_list, weight_names):
        if trained:
            model.set_weights(weights)
            metrics_lists[weight_name] = []
            for dataset, dataset_name in zip(datasets, dataset_names):
                metricsi = get_model_losses(
                    model, 
                    dataset = dataset, 
                    tuple_dataset = tuple_dataset, 
                    get_metrics = True, 
                    brier = survival_evaluation_brier, 
                    brier_times = survival_evaluation_brier_times, 
                    manual_batch_flag = False)
                _ = metricsi.pop('reconstruction_loss_dataset2', None)
                _ = metricsi.pop('batch_cross_entropy', None)
                _ = metricsi.pop('batch_dsc', None)
                drug_response_r2 = metricsi.pop('drug_response_r2_scores', None)
                if drug_response_r2 is not None:
                    drug_metricsi = dict(zip(
                        [f"drug{i}_r2" for i in range(len(drug_response_r2))],
                        drug_response_r2))
                    metricsi = {**metricsi, **drug_metricsi}
                metrics_df = pd.DataFrame(metricsi, index = [0])
                metrics_df.columns = [
                    re.sub('reconstruction_loss_dataset1', 'reconstruction_loss', i) 
                    for i in metrics_df.columns]
                metrics_df.columns = [dataset_name + '_' + i for i in metrics_df.columns]
                metrics_lists[weight_name].append(metrics_df)
            for dataset_pair, dataset_pair_name in zip(dataset_pairs, dataset_pair_names):
                z1 = get_embeddings(model, dataset_pair[0], tuple_dataset = tuple_dataset)
                z2 = get_embeddings(model, dataset_pair[1], tuple_dataset = tuple_dataset)
                z = tf.concat((z1, z2), axis = 0)
                if model.deconfounder_model:
                    shared_embedding_mask = tf.math.reduce_all(
                        model.batch_specific_layer_mask > 0, axis = 0)
                    z = tf.boolean_mask(z, shared_embedding_mask, axis = 1)
                b = tf.concat((tf.constant(0, shape = (z1.shape[0],)), 
                               tf.constant(1, shape = (z2.shape[0],))), 
                              axis = 0)
                batch_metrics = {'batch_loss' : model.batch_correction_loss(z, b).numpy(), 
                                 'batch_dsc' : DSC(z, b).numpy()}
                metrics_df = pd.DataFrame(batch_metrics, index = [0])
                metrics_df.columns = [dataset_pair_name + '_' + i for i in metrics_df.columns]
                metrics_lists[weight_name].append(metrics_df)
            metrics_lists[weight_name] = pd.concat(metrics_lists[weight_name], axis = 1)
            metrics_lists[weight_name]['stage'] = weight_name
    metrics_df = pd.concat(metrics_lists, axis = 0)
    metrics_df['task'] = task_id
    metrics_df['run'] = run
    metrics_df['fold'] = fold
    metrics_df.to_csv(
        file_name_prefix + 'metrics_run{}_fold{}_task{}.csv'.format(run, fold, task_id), 
        na_rep = 'NA', 
        header = True, 
        index = False)
    
    diagnostics_keys = [
        'train_losses_ae', 
        'valid_losses_ae',
        'train_losses_cl',
        'valid_losses_cl', 
        'train_losses_sr',
        'valid_losses_sr', 
        'train_losses_dr',
        'valid_losses_dr', 
        'train_losses_bd',
        'valid_losses_bd', 
        'train_losses_bc',
        'valid_losses_bc', 
        'train_losses', 
        'valid_losses']
    diagnostics = [result[i] for i in diagnostics_keys if result.get(i, None) is not None]
    diagnostics_stage = [i for i in diagnostics_keys if result.get(i, None) is not None]
    for d, s in zip(diagnostics, diagnostics_stage):
        d['stage'] = s
    if len(diagnostics) > 0:
        diagnostics = pd.concat(diagnostics, axis = 0)
        diagnostics['task'] = task_id
        diagnostics['run'] = run
        diagnostics['fold'] = fold
        diagnostics.to_csv(
            file_name_prefix + 'diagnostics_run{}_fold{}_task{}.csv.gz'.format(
                run, fold, task_id), 
            na_rep = 'NA', 
            header = True, 
            index = False)
    
    if False and pre_trained_model_embeddings:
        phases_of_interest = ['final', 'ae_pt', 'bc_pt']
    else:
        phases_of_interest = ['final']
    prediction_model = (model.supervised or model.survival_model or model.drug_response_model)
    for trained, weights, weight_name in zip(trained_list, weight_list, weight_names):
        if trained and weight_name in phases_of_interest:
            model.set_weights(weights)
            embeddings = []
            predictions = []
            for dataset, rows, dataset_name in zip(datasets, dataset_rows, dataset_names):
                zi = get_embeddings(model, dataset, tuple_dataset = tuple_dataset)
                zi = pd.DataFrame(
                    zi, 
                    index = rows,
                    columns = ['z{}'.format(i+1) for i in range(zi.shape[1])])
                zi['dataset'] = dataset_name
                embeddings.append(zi)
                if weight_name == 'final' and prediction_model:
                    pi = get_model_predictions(model, dataset, tuple_dataset = tuple_dataset)
                    for pred, key in zip(pi.values(), pi.keys()):
                        if pred.shape[1]:
                            cols = [key + '_' + str(i) for i in np.arange(pred.shape[1])]
                        else:
                            cols = key
                        pi[key] = pd.DataFrame(pi[key], index = rows, columns = cols)
                    pi = pd.concat(pi.values(), axis = 1)
                    pi['dataset'] = dataset_name
                    predictions.append(pi)
            embeddings = pd.concat(embeddings, axis = 0)
            embeddings['task'] = task_id
            embeddings['run'] = run
            embeddings['fold'] = fold
            embeddings.to_csv(
                f"{file_name_prefix}embeddings_run{run}_fold{fold}_task{task_id}.csv.gz", 
                na_rep = 'NA', 
                header = True, 
                index = True)
            
            if weight_name == 'final' and prediction_model:
                predictions = pd.concat(predictions, axis = 0)
                predictions['task'] = task_id
                predictions['run'] = run
                predictions['fold'] = fold
                predictions.to_csv(
                    f"{file_name_prefix}predictions_run{run}_fold{fold}_task{task_id}.csv.gz", 
                    na_rep = 'NA', 
                    header = True, 
                    index = True)
    
    return

def get_classifier_metrics(
    model, 
    z_mean, 
    y, 
    labels
):
    y_prob = softmax(model.classifier_net(z_mean), axis = 1)
    y_pred = np.argmax(y_prob, axis = 1)
    y_not_missing = y != -1
    metrics = pd.DataFrame.from_dict({})
    metrics['accuracy'] = [skl_metrics.accuracy_score(y[y_not_missing], y_pred[y_not_missing])]
    metrics['auroc'] = skl_metrics.roc_auc_score(
        y[y_not_missing], 
        y_prob[y_not_missing,:],
        average = 'macro', 
        multi_class = 'ovr',
        labels = labels)
    metrics['f1'] = skl_metrics.f1_score(
        y[y_not_missing], 
        y_pred[y_not_missing],
        average = 'macro', 
        labels = labels)
    metrics['balanced_accuracy'] = skl_metrics.balanced_accuracy_score(
        y[y_not_missing], 
        y_pred[y_not_missing])
    return metrics

def get_survival_eval_vars(
    model, 
    z, 
    time, 
    event, 
    covariates = None, 
    mask = None
):
    if mask is None:
        mask = time >= 0
    z = tf.boolean_mask(z, mask, axis = 0)
    time = tf.boolean_mask(time, mask, axis = 0)
    event = tf.boolean_mask(event, mask, axis = 0)
    if covariates is None:
        z_a = z
    else:
        covariates = tf.boolean_mask(covariates, mask, axis = 0)
        z_a = tf.concat([z, covariates], axis = 1)
    
    hazard_components = model.hazard_component_net(z_a)
    risk = model.survival_risk_net(hazard_components)[:,0]
    
    return {'risk' : risk, 
            'time' : time, 
            'event' : event}

'''
Get survival metrics

Integrated Brier score requires time predictions. Whe using a Cox PH model, 
a Kaplan-Meier model must be used for the baseline survival. 
To avoid data leakage, the baseline should be fitted on training data only. 
'''
def get_survival_metrics(
    risk, 
    time, 
    event, 
    risk_train = None,
    time_train = None, 
    event_train = None, 
    brier_times = None, 
    brier = False
):
    if integrated_brier_score:
        if risk_train is None or time_train is None or event_train is None:
            risk_train = risk
            time_train = time
            event_train = event
        if brier_times is None:
            brier_times = np.arange(np.min(time[event]), np.max(time[event])+1)
    
    metrics = {}
    metrics['surv_c'] = concordance_index_censored(event, time, risk)[0]
    if brier:
        test_struct = np.array([(event[i] == 1, time[i]) for i in range(event.shape[0])],
                               dtype = [('event', '?'),('time', '<i4')])
        train_struct = np.array(
            [(event_train[i] == 1, time_train[i]) for i in range(event_train.shape[0])],
            dtype = [('event', '?'),('time', '<i4')])
        
        baseline_survival = breslow_baseline_survival_time_estimate(
            event_train, time_train, risk_train, brier_times)
        surv_proba = np.array([np.power(bs, np.exp(risk)) for bs in baseline_survival]).T
        
        metrics['brier_score'] = integrated_brier_score(
            train_struct, test_struct, surv_proba, brier_times)
        metrics['brier_start'] = np.min(brier_times)
        metrics['brier_end'] = np.max(brier_times)
    
    return metrics

def compute_supervised_metrics(model, y_logit, y, mask, acc, auroc, aupr):
    # For supervised loss only use non-missing labels
    y_nm = tf.boolean_mask(y, mask, axis = 0)
    y_onehot = tf.one_hot(y_nm, model.class_number)
    y_logit_nm = tf.boolean_mask(y_logit, mask, axis = 0)
    acc.update_state(y_onehot, y_logit_nm)
    auroc.update_state(y_onehot, y_logit_nm)
    aupr.update_state(y_onehot, y_logit_nm)
    # No need for return value
    return
