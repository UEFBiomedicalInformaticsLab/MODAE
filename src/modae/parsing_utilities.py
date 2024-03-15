# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 11:10:53 2022

@author: rintala
"""

import numpy as np
import tensorflow as tf

def str_list_parser(input_str):
    if len(input_str.strip()):
        return input_str.split(',')
    return []

def str_parser(input_str):
    if len(input_str.strip()):
        return input_str
    return None

def int_parser(input_int):
    if input_int >= 0:
        return input_int
    return None

def int_range_parser(input_str):
    out_range = [int(i) for i in input_str.split('-')]
    if len(out_range) > 1:
        out_min = np.min(out_range)
        out_max = np.max(out_range)
        out = np.random.randint(out_min, out_max + 1)
    else:
        out = int(out_range[0])
    return out

def parameter_range_parser(input_str, log_scale = False):
    range_out = input_str.split('-')
    range_out = [float(i) for i in range_out]
    if len(range_out) == 1:
        generator_out = range_out[0]
    else:
        if log_scale:
            generator_out = np.exp(np.log(range_out[0]) + \
                            np.random.uniform(size = 1).item() * \
                            (np.log(range_out[1]) - np.log(range_out[0])))
        else:
            generator_out = range_out[0] + \
                            np.random.uniform(size = 1).item() * \
                            (range_out[1] - range_out[0])
    
    return generator_out

def layer_parser(input_str):
    out = []
    layers = input_str.split(',')
    for layer in layers:
        out.append(int_range_parser(layer))
    return out

def iterator_range_parser(input_str):
    if (input_str):
        range_out = input_str.split(':')
        return np.arange(int(range_out[0]), int(range_out[1]) + 1)
    else:
        return None

## Objective order (hard-coded in autoencoder_model.py)
# 1) reconstruction MSE (used for both patient tumors and cell-lines)
# 2) classifier cross-entroy
# 3) survival Cox-PH likelihood
# 4) adversarial loss, CE for GAN, critic loss for WGAN
# 5) drug-response MSE
def objective_ratio_parser(input_str, norm_ord = 1., input_ranges = True):
    out = []
    ranges = input_str.split(',')
    for range_i in ranges:
        if input_ranges:
            bounds = range_i.split('-')
            if len(bounds) > 1:
                out.append(np.random.uniform(size = 1).item() * (float(bounds[1]) - float(bounds[0])) + float(bounds[0]))
            else:
                out.append(float(bounds[0]))
        else:
            out.append(float(range_i))
    return (out / np.linalg.norm(out, ord = norm_ord))

# Parameter generator
def search_arg_generator(
        args, 
        data_dict, 
        task_id = 0, 
        namespace = True, 
        objective_weight_range_input = True): 
    if namespace:
        args_dict = vars(args)
    else:
        # Assume input is dict instead
        args_dict = args
    '''VAE parameter search parameter generator
    
    Used to create input parameters for the random_parameter_search utility 
    function. 
    '''
    # TODO: change once multi-view is enabled
    omics_layer = 'mrna'
    
    if len(args_dict.get('classifier_layers', '')):
        classifier_layers = layer_parser(args_dict.get('classifier_layers'))
    else: 
        classifier_layers = []
    if len(args_dict.get('survival_model_layers', '')):
        survival_model_layers = layer_parser(args_dict.get('survival_model_layers'))
    else: 
        survival_model_layers = []
    if len(args_dict.get('batch_detector_layers', '')):
        batch_adversarial_model_layers = layer_parser(args_dict.get('batch_detector_layers'))
    else: 
        batch_adversarial_model_layers = []
    if len(args_dict.get('drug_response_model_layers', '')):
        drug_response_model_layers = layer_parser(args_dict.get('drug_response_model_layers'))
    else: 
        drug_response_model_layers = []
    if len(args_dict.get('drug_response_model_drugwise_layers', '')):
        drug_response_model_drugwise_layers = layer_parser(args_dict.get('drug_response_model_drugwise_layers'))
    else: 
        drug_response_model_drugwise_layers = []
    
    
    # Parameters passed to VAE()
    model_args = {
        'encoder_layers' : layer_parser(args_dict.get('encoder_layers')),
        'decoder_layers' : layer_parser(args_dict.get('decoder_layers')),
        'reg_a' : parameter_range_parser(args_dict.get('reg_weights', '0.001'), log_scale = True), 
        'fix_var' : args_dict.get('mse', True),
        'variational' : args_dict.get('variational', False),
        'noise_sd' : parameter_range_parser(args_dict.get('reg_noise_sd', '1.')), 
        'dropout_input' : parameter_range_parser(args_dict.get('reg_dropout_rate_input', '0.')), 
        'dropout_autoencoder' : parameter_range_parser(args_dict.get('reg_dropout_rate_autoencoder', '0.')), 
        'dropout_survival' : parameter_range_parser(args_dict.get('reg_dropout_rate_survival', '0.')), 
        'dropout_classifier' : parameter_range_parser(args_dict.get('reg_dropout_rate_classifier', '0.')), 
        'dropout_batch' : parameter_range_parser(args_dict.get('reg_dropout_rate_batch', '0.')), 
        'dropout_drug_response' : parameter_range_parser(args_dict.get('reg_dropout_rate_drug_response', '0.')), 
        'reg_type' : args_dict.get('reg_weights_type', 'L2'),
        'supervised' : args_dict.get('supervised', False), 
        'classifier_layers' : classifier_layers,
        'survival_model' : args_dict.get('survival_model', False),
        'survival_model_layers' : survival_model_layers,
        'batch_adversarial_model' : args_dict.get('batch_correction', False), 
        'batch_adversarial_model_layers' : batch_adversarial_model_layers, 
        'batch_adversarial_loss_function' : args_dict.get('batch_loss', 'wasserstein'), 
        'deconfounder_layers_per_batch' : int_range_parser(args_dict.get('deconfounder_layers_per_batch', '0')), 
        'deconfounder_norm_penalty' : parameter_range_parser(args_dict.get('deconfounder_norm_penalty', '0.')), 
        'deconfounder_centered_alignment' : args_dict.get('deconfounder_centered_alignment', False), 
        'batch_adversarial_gradient_penalty' : parameter_range_parser(
            args_dict.get('batch_adversarial_gradient_penalty', '0.'), 
            log_scale = True), 
        'drug_response_model' : args_dict.get('drug_response_model', False), 
        'drug_response_model_output_activation' : str_parser(args_dict.get('drug_response_model_output_activation', None)), 
        'drug_response_model_layers' : drug_response_model_layers, 
        'drug_response_model_drugwise_layers' : drug_response_model_drugwise_layers, 
        'objective_weights' : objective_ratio_parser(
            args_dict.get('objective_weights', '1,1,1,1,1'), 
            args_dict.get('objective_weight_norm_order', 1.), 
            input_ranges = objective_weight_range_input), 
        'hidden_activation' : args_dict.get('hidden_activation', 'relu')
        }
    if args_dict.get('hidden_activation', 'relu') == 'selu':
        model_args['hidden_initializer_function'] = tf.keras.initializers.LecunUniform
        model_args['hidden_dropout_function'] = tf.keras.layers.AlphaDropout
    model_args['input_noise'] = True if model_args['noise_sd'] > 0.0 else False
    
    # Parameters passed to dataset_batch_setup()
    data_batch_args = {
        'cache' : args_dict.get('cache_data', False),
        'prefetch' : args_dict.get('prefetch_data', False),
        'shuffle' : not args_dict.get('no_minibatches', False),
        'mini_batches' : not args_dict.get('no_minibatches', True),
        'batch_size' : args_dict.get('minibatchsize', 128)
        }
    
    # Parameters passed to train()
    train_args = {
        'early_stopping' : args_dict.get('earlystop', False),
        'print_period' : args_dict.get('print_period', 1),
        'patience' : parameter_range_parser(args_dict.get('patience', '100'), log_scale = True), 
        'return_losses' :  not args_dict.get('no_diagnostics', False), # TODO: any case you would not want diagnostics?
        'supervised_metric' : args_dict.get('supervised_metric', 'cross_entropy'), 
        'debug' : args_dict.get('debug_training', False)
        }
    
    adam_args = {
        'beta_1' : parameter_range_parser(args_dict.get('adam_beta1'), log_scale = True), 
        'beta_2' : parameter_range_parser(args_dict.get('adam_beta2'), log_scale = True)
    }
    if args_dict.get('adversarial_pre_training_optimizer') == 'adam':
        adversarial_optimizer_args = adam_args
        adversarial_optimizer_class = tf.keras.optimizers.Adam
    else:
        adversarial_optimizer_args = {}
        adversarial_optimizer_class = tf.keras.optimizers.experimental.SGD
    if args_dict.get('batch_correction_pre_training_optimizer') == 'adam':
        batch_correction_optimizer_args = adam_args
        batch_correction_optimizer_class = tf.keras.optimizers.Adam
    else:
        batch_correction_optimizer_args = {}
        batch_correction_optimizer_class = tf.keras.optimizers.experimental.SGD
    
    # Parameters passed to train_aecl_with_pretraining()
    gym_args = {
        'max_epochs' : args_dict.get('epochs', 100),
        'max_epochs_pre_ae' : args_dict.get('epochs_pre_ae', 100),
        'max_epochs_pre_cl' : args_dict.get('epochs_pre_cl', 100),
        'max_epochs_pre_sr' : args_dict.get('epochs_pre_sr', 100),
        'max_epochs_pre_bd' : args_dict.get('epochs_pre_bd', 100), 
        'max_epochs_pre_bc' : args_dict.get('epochs_pre_bc', 100), 
        'max_epochs_pre_dr' : args_dict.get('epochs_pre_dr', 100), 
        'learning_rate' : parameter_range_parser(args_dict.get('learning_rate', '0.0001'), log_scale = True),
        'pre_learning_rate_ae' : parameter_range_parser(args_dict.get('pre_learning_rate_ae', '0.001'), log_scale = True), 
        'pre_learning_rate_cl' : parameter_range_parser(args_dict.get('pre_learning_rate_cl', '0.001'), log_scale = True), 
        'pre_learning_rate_sr' : parameter_range_parser(args_dict.get('pre_learning_rate_sr', '0.001'), log_scale = True), 
        'pre_learning_rate_bd' : parameter_range_parser(args_dict.get('pre_learning_rate_bd', '0.001'), log_scale = True), 
        'pre_learning_rate_bc' : parameter_range_parser(args_dict.get('pre_learning_rate_bc', '0.001'), log_scale = True), 
        'pre_learning_rate_dr' : parameter_range_parser(args_dict.get('pre_learning_rate_dr', '0.001'), log_scale = True), 
        'pre_train_ae' : not args_dict.get('no_pre_train_ae', False), 
        'pre_train_cl' : not args_dict.get('no_pre_train_cl', False), 
        'pre_train_sr' : not args_dict.get('no_pre_train_sr', False), 
        'pre_train_bd' : not args_dict.get('no_pre_train_bd', False), 
        'pre_train_bc' : not args_dict.get('no_pre_train_bc', False), 
        'pre_train_dr' : not args_dict.get('no_pre_train_dr', False), 
        'optimizer_args' : adam_args, 
        'adversarial_optimizer_class' : adversarial_optimizer_class, 
        'adversarial_optimizer_args' : adversarial_optimizer_args, 
        'correction_optimizer_class' : batch_correction_optimizer_class, 
        'correction_optimizer_args' : batch_correction_optimizer_args, 
        'early_stopping_set_denominator' : args_dict.get('early_stopping_set_denominator', 10), # = number of folds of which one is used
        'early_stopping_set_stratify' : True,
        'debug' : args_dict.get('debug_gym', False)
        }
    gym_args['adversarial_learning_rate'] = gym_args['learning_rate'] * \
        parameter_range_parser(args_dict.get('adversarial_learning_rate_multi', '1.0'), log_scale = True)
    
    # Parameters passed to parameter_search_iter()
    out = {
        'data_dict' : data_dict, 
        'nruns' : args_dict.get('cv_runs', 1),
        'nfolds' : args_dict.get('cv_folds', 5),
        'model_args' : model_args,
        'train_args' : train_args,
        'gym_args' : gym_args,
        'data_batch_args' : data_batch_args,
        'data_standardize' : args_dict.get('data_standardize', False),
        'task_id' : task_id,
        'save' : True,
        'file_name_prefix' : args_dict.get('p') + args_dict.get('o'),
        'omics_layer' : omics_layer, 
        'parallel' : args_dict.get('parallel', False),
        'gpu_memory' : args_dict.get('gpu_memory', -1) if args_dict.get('gpu_memory', -1) > 0 else None,
        'nthreads' : args_dict.get('nthreads', 4),
        'nthreads_interop' : args_dict.get('nthreads_interop', 2),
        'survival_evaluation_brier_times' : iterator_range_parser(args_dict.get('survival_evaluation_brier_times', '')),
        'ps_validation_sets' : not args_dict.get('no_ps_validation', False), 
        'ps_test_sets' : not args_dict.get('no_ps_test', False)
        }
    return out

