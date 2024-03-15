# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 11:10:53 2022

@author: rintala
"""

import argparse
#import tensorflow as tf
import numpy as np
import pandas as pd
import json
#import time
import os
#import sys
#import rdata
#from scipy.special import softmax
#import logging


from modae.training_utilities import parameter_search_iter, parallel_parameter_search_iter, serialized_parameter_search_iter
from modae.data_utilities import complete_data_loader, data_serialization, FeatureSpecEncoder
from modae.parsing_utilities import search_arg_generator, str_list_parser, str_parser, int_parser


#%% Main
def main(args):
    # Load data
    if args.serialize_data:
        if args.data_setup_only:
            data_dict = complete_data_loader(
                patient_expression_root_dir = str_parser(args.patient_expression_root_dir),
                patient_expression_cancer_list = str_list_parser(args.patient_expression_cancer_list), 
                patient_expression_cancer_subset = str_list_parser(args.patient_expression_cancer_subset), 
                patient_expression_file = str_parser(args.patient_expression_file_name),
                patient_expression_sample_file = str_parser(args.patient_expression_sample_file),
                patient_expression_sample_id_col = str_parser(args.patient_expression_sample_id_col),
                patient_expression_sample_type_col = str_parser(args.patient_expression_sample_type_col),
                patient_expression_sample_type_include = str_list_parser(args.patient_expression_sample_type_include),
                patient_gene_mapping_file = str_parser(args.patient_gene_mapping_file),
                patient_expression_log2 = not args.no_patient_expression_log2, 
                patient_expression_mean_cut = args.patient_expression_mean_cut, 
                patient_expression_standardize_early = args.patient_expression_standardize_early, 
                patient_expression_filter_file = str_parser(args.patient_expression_filter_file), 
                patient_survival_file = str_parser(args.patient_survival_file),
                patient_survival_event_col = str_parser(args.patient_survival_event_col),
                patient_survival_time_col = str_parser(args.patient_survival_time_col),
                patient_survival_covar_cols = str_list_parser(args.patient_survival_covar_cols), 
                patient_survival_covar_onehot = [bool(int(i)) for i in str_list_parser(args.patient_survival_covar_onehot)], 
                patient_id_col = args.patient_id_col, 
                patient_id_tcga_barcode = not args.no_patient_id_tcga_barcode, 
                patient_redaction_col = args.patient_redaction_col, 
                cell_line_expression_root_dir = str_parser(args.cell_line_expression_root_dir),
                cell_line_expression_file = str_parser(args.cell_line_expression_file),
                cell_line_gene_mapping_file = str_parser(args.cell_line_gene_mapping_file),
                cell_line_expression_mean_cut = args.cell_line_expression_mean_cut, 
                cell_line_expression_standardize_early = args.cell_line_expression_standardize_early, 
                cell_line_expression_filter_file = str_parser(args.cell_line_expression_filter_file), 
                cell_line_filter_mapping_file = str_parser(args.cell_line_filter_mapping_file),
                cell_line_filter_mapping_column = str_parser(args.cell_line_filter_mapping_column),
                cell_line_filter_inclusion_list = str_list_parser(args.cell_line_filter_inclusion_list), 
                cell_line_filter_exclusion_list = str_list_parser(args.cell_line_filter_exclusion_list), 
                gene_harmonization_union = not args.no_gene_harmonization_union, 
                cell_line_drug_response_root_dir = str_parser(args.cell_line_drug_response_root_dir),
                cell_line_drug_response_file = str_parser(args.cell_line_drug_response_file),
                cell_line_drug_response_row_info_file = str_parser(args.cell_line_drug_response_row_info_file),
                cell_line_drug_response_row_info_treat_col = str_parser(args.cell_line_drug_response_row_info_treat_col),
                cell_line_drug_response_row_map_file = str_parser(args.cell_line_drug_response_row_map_file),
                cell_line_drug_response_target_column = str_parser(args.cell_line_drug_response_target_column),
                cell_line_drug_response_maxscale = args.cell_line_drug_response_maxscale, 
                shuffle = not args.no_complete_data_shuffle, 
                shuffle_seed = int_parser(args.complete_data_shuffle_seed)
            )
            if args.verbose:
                print('Loaded data contains keys: ' + str(data_dict.keys()))
        else:
            data_dict = None
    else:
        raise ValueError('Data must be serialized in the current version of MODAE.')
    
    # Use slurm array task to seed hyper-parameter values
    slurm_array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if slurm_array_task_id:
        slurm_array_task_id = int(slurm_array_task_id)
        np.random.seed(slurm_array_task_id)
    else:
        slurm_array_task_id = None
    
    os.makedirs(args.p, exist_ok=True) # 
    
    search_kwargs = search_arg_generator(
        args = args, 
        data_dict = data_dict,
        task_id = slurm_array_task_id)
    
    if args.serialize_data:
        if args.data_setup_only:
            '''
            Prepare splits into TFRecords files
            This includes pre-processing for each CV fold
            '''
            serialized_data = data_serialization(
                data_dict = data_dict, 
                nruns = search_kwargs.get('nruns', 1), 
                nfolds = search_kwargs.get('nfolds', 5), 
                stratify_survival = vars(args).get('stratify_survival', True), 
                stratify_subtype = vars(args).get('stratify_subtype', True), 
                stratify_dr = vars(args).get('stratify_dr', True), 
                stratify_time_quantiles = vars(args).get('stratify_time_quantiles', 10), 
                cv_seed = vars(args).get('cv_seed', 0), 
                model_args = search_kwargs.get('model_args', {}), 
                data_standardize = search_kwargs.get('data_standardize', False), 
                file_name_prefix = search_kwargs.get('file_name_prefix', ''), 
                omics_layer = search_kwargs.get('omics_layer', 'mrna'), 
                ps_validation_sets = search_kwargs.get('ps_validation_sets', True), 
                ps_test_sets = search_kwargs.get('ps_test_sets', True))
            fn = search_kwargs['file_name_prefix'] + '_serialized_data_spec.json'
            handle = open(fn, 'w')
            handle.write(json.dumps(FeatureSpecEncoder(serialized_data), cls = json.JSONEncoder))
            handle.close()
            if args.verbose:
                print('Serialized data spec saved to: ' + fn)
            if args.debug_data:
                for k, v in data_dict.items():
                    print(f"{k} shape : {v.shape}")
            return 0
        else:
            handle = open(search_kwargs['file_name_prefix'] + '_serialized_data_spec.json', 'r')
            search_kwargs['serialized_data'] = json.load(handle)
            handle.close()
    
    # Save parameter settings
    pandas_table_m = [(k, [str(v)]) for k,v in search_kwargs['model_args'].items()]
    pandas_table_t = [(k, [str(v)]) for k,v in search_kwargs['train_args'].items()]
    pandas_table_g = [(k, [str(v)]) for k,v in search_kwargs['gym_args'].items()]
    pandas_table_b = [(k, [str(v)]) for k,v in search_kwargs['data_batch_args'].items()]
    pandas_table_i = [(k, [str(v)]) for k,v in search_kwargs.items()]
    
    pandas_table = (
        pandas_table_m + 
        pandas_table_t + 
        pandas_table_g + 
        pandas_table_b +
        pandas_table_i)
    pandas_table = dict(pandas_table)
    _ = pandas_table.pop('x', None)
    _ = pandas_table.pop('y', None)
    _ = pandas_table.pop('ind', None)
    _ = pandas_table.pop('model_args', None)
    _ = pandas_table.pop('train_args', None)
    _ = pandas_table.pop('gym_args', None)
    _ = pandas_table.pop('data_batch_args', None)
    
    if args.survival_model:
        # remove survival data from parameter table
        _ = pandas_table.pop('survival_event', None)
        _ = pandas_table.pop('survival_time', None)
        _ = pandas_table.pop('survival_covariates', None)
    
    pandas_table = pd.DataFrame.from_dict(pandas_table)
    pandas_table.to_csv(search_kwargs['file_name_prefix'] + 'parameters_task{}.csv'.format(search_kwargs['task_id']),
        na_rep = 'NA', header = True, index = False)
    
    # Run search iteration
    if args.parallel:
        if args.ptimeout == 0:
            args.ptimeout = np.inf
        return parallel_parameter_search_iter(
            max_cv_repeats = args.max_cv_repeats, 
            process_timeout = args.ptimeout,
            nprocess = args.nprocess,
            sleep_time = 5,
            iter_args = search_kwargs,
            serialize_data = args.serialize_data)
    else:
        if args.serialize_data:
            _ = search_kwargs.pop('data_dict', None)
            _ = search_kwargs.pop('ps_validation_sets', None)
            _ = search_kwargs.pop('ps_test_sets', None)
            return serialized_parameter_search_iter(**search_kwargs)
        else:
            return parameter_search_iter(**search_kwargs)

if __name__ == '__main__':
    desc_str = 'Command line random parameter search iteration tool. \
    \
    Runs one iteration of random parameter search using the given generator \
    settings. '
    #%% Parse command line arguments
    parser = argparse.ArgumentParser(prog = 'RPSiter', description = desc_str)
    parser.add_argument('-x', type = str, default = '') # input rdata file path
    parser.add_argument('-l', '--omics_layers', type = str, default = 'mrna')
    parser.add_argument('-o', type = str, default = '') # output file prefix
    parser.add_argument('-p', type = str, default = './') # output path
    parser.add_argument('--verbose', action = 'store_true')
    parser.add_argument('--no_diagnostics', action = 'store_true')
    
    #parser.add_argument('--rseed', type=int)
    parser.add_argument('--max_cv_repeats', type=int, default = np.inf)
    parser.add_argument('--cv_runs', type=int, default = 1)
    parser.add_argument('--cv_folds', type=int, default = 5)
    parser.add_argument('--no_ps_validation', action = 'store_true')
    parser.add_argument('--no_ps_test', action = 'store_true')
    
    parser.add_argument('--no_minibatches', action = 'store_true')
    parser.add_argument('--minibatchsize', type=int, default = 128)
    
    # General architecture
    parser.add_argument('--hidden_activation', type = str, default = 'relu')
    parser.add_argument('--objective_weights', type = str, default = '1.,1.,1.,1.,1.,1.')
    parser.add_argument('--objective_weight_norm_order', type = float, default = 1.)
    
    # Autoencoder architecture
    parser.add_argument('--mse', action = 'store_true')
    parser.add_argument('--variational', action = 'store_true')
    parser.add_argument('--encoder_layers', type = str, default = '128,64,10')
    parser.add_argument('--decoder_layers', type = str, default = '64,128')
    
    # Classifier architecture
    parser.add_argument('--supervised', action = 'store_true')
    parser.add_argument('--classifier_layers', type = str, default = '')
    parser.add_argument('--supervised_metric', type = str, default = 'cross_entropy') # for early stop
    
    # Training arguments
    parser.add_argument('--learning_rate', type = str, default = '0.00001')
    parser.add_argument('--adversarial_learning_rate_multi', type = str, default = '1.0')
    parser.add_argument('--pre_learning_rate_ae', type = str, default = '0.001')
    parser.add_argument('--pre_learning_rate_cl', type = str, default = '0.001')
    parser.add_argument('--pre_learning_rate_sr', type = str, default = '0.001')
    parser.add_argument('--pre_learning_rate_bd', type = str, default = '0.001')
    parser.add_argument('--pre_learning_rate_bc', type = str, default = '0.001')
    parser.add_argument('--pre_learning_rate_dr', type = str, default = '0.001')
    parser.add_argument('--no_pre_train_ae', action = 'store_true')
    parser.add_argument('--no_pre_train_cl', action = 'store_true')
    parser.add_argument('--no_pre_train_sr', action = 'store_true')
    parser.add_argument('--no_pre_train_bd', action = 'store_true')
    parser.add_argument('--no_pre_train_bc', action = 'store_true')
    parser.add_argument('--no_pre_train_dr', action = 'store_true')
    parser.add_argument('--adversarial_pre_training_optimizer', type = str, default = 'sgd')
    parser.add_argument('--batch_correction_pre_training_optimizer', type = str, default = 'sgd')
    
    parser.add_argument('--epochs', type=int, default = 1000)
    parser.add_argument('--epochs_pre_ae', type=int, default = 100)
    parser.add_argument('--epochs_pre_cl', type=int, default = 100)
    parser.add_argument('--epochs_pre_sr', type=int, default = 100)
    parser.add_argument('--epochs_pre_bd', type=int, default = 100)
    parser.add_argument('--epochs_pre_bc', type=int, default = 100)
    parser.add_argument('--epochs_pre_dr', type=int, default = 100)
    parser.add_argument('--earlystop', action = 'store_true')
    parser.add_argument('--print_period', type=int, default = 1)
    parser.add_argument('--early_stopping_set_denominator', type=int, default = 10)
    parser.add_argument('--patience', type = str, default = '100')
    
    parser.add_argument('--reg_weights_type', default = 'L2')
    parser.add_argument('--reg_weights', type = str, default = '0.0001-1.0')
    parser.add_argument('--reg_noise_sd', type = str, default = '0.')
    parser.add_argument('--reg_dropout_rate_input', type = str, default = '0.')
    parser.add_argument('--reg_dropout_rate_autoencoder', type = str, default = '0.')
    parser.add_argument('--reg_dropout_rate_survival', type = str, default = '0.')
    parser.add_argument('--reg_dropout_rate_classifier', type = str, default = '0.')
    parser.add_argument('--reg_dropout_rate_batch', type = str, default = '0.')
    parser.add_argument('--reg_dropout_rate_drug_response', type = str, default = '0.')
    
    # Data arguments
    parser.add_argument('--data_var_cutoff_rank', type = int, default = -1)
    parser.add_argument('--data_label_column', type = str, default = 'subtype')
    parser.add_argument('--data_standardize', action = 'store_true')
    
    parser.add_argument('--survival_model', action = 'store_true')
    parser.add_argument('--survival_model_layers', type = str, default = '')
    parser.add_argument('--survival_evaluation_brier_times', type = str, default = '')
    
    parser.add_argument('--batch_correction', action = 'store_true')
    parser.add_argument('--batch_detector_layers', type = str, default = '')
    parser.add_argument('--batch_loss', type = str, default = 'wasserstein')
    parser.add_argument('--deconfounder_layers_per_batch', type = str, default = '0')
    parser.add_argument('--deconfounder_norm_penalty', type = str, default = '0.')
    parser.add_argument('--deconfounder_centered_alignment', action = 'store_true')
    parser.add_argument('--batch_adversarial_gradient_penalty', type = str, default = '0.')
    parser.add_argument('--adam_beta1', type = str, default = '0.9')
    parser.add_argument('--adam_beta2', type = str, default = '0.999')
    
    parser.add_argument('--drug_response_model', action = 'store_true')
    parser.add_argument('--drug_response_model_output_activation', type = str, default = '')
    parser.add_argument('--drug_response_model_layers', type = str, default = '')
    parser.add_argument('--drug_response_model_drugwise_layers', type = str, default = '')
    
    parser.add_argument('--parallel', action = 'store_true')
    parser.add_argument('--ptimeout', type = int, default = 0)
    parser.add_argument('--nprocess', type = int, default = 1)
    parser.add_argument('--nthreads', type = int, default = 4)
    parser.add_argument('--nthreads_interop', type = int, default = 2)
    parser.add_argument('--gpu_memory', type = int, default = -1)
    
    # Serialized complete data loader arguments
    parser.add_argument('--serialize_data', action = 'store_true')
    parser.add_argument('--data_setup_only', action = 'store_true')
    parser.add_argument('--patient_expression_root_dir', type = str, default = '')
    parser.add_argument('--patient_expression_cancer_list', type = str, default = '')
    parser.add_argument('--patient_expression_cancer_subset', type = str, default = '')
    parser.add_argument('--patient_expression_file_name', type = str, default = None)
    parser.add_argument('--patient_expression_sample_file', type = str, default = '')
    parser.add_argument('--patient_expression_sample_id_col', type = str, default = 'colname')
    parser.add_argument('--patient_expression_sample_type_col', type = str, default = 'type')
    parser.add_argument('--patient_expression_sample_type_include', type = str, default = '1')
    parser.add_argument('--patient_gene_mapping_file', type = str, default = '')
    parser.add_argument('--no_patient_expression_log2', action = 'store_true')
    parser.add_argument('--patient_expression_mean_cut', type = float, default = -1.)
    parser.add_argument('--patient_expression_standardize_early', action = 'store_true')
    parser.add_argument('--patient_expression_filter_file', type = str, default = '')
    parser.add_argument('--patient_survival_file', type = str, default = '')
    parser.add_argument('--patient_survival_event_col', type = str, default = '')
    parser.add_argument('--patient_survival_time_col', type = str, default = '')
    parser.add_argument('--patient_survival_covar_cols', type = str, default = '')
    parser.add_argument('--patient_survival_covar_onehot', type = str, default = '')
    
    parser.add_argument('--patient_id_col', type = str, default = '')
    parser.add_argument('--no_patient_id_tcga_barcode', action = 'store_true')
    parser.add_argument('--patient_redaction_col', type = str, default = '')
    
    parser.add_argument('--cell_line_expression_root_dir', type = str, default = '')
    parser.add_argument('--cell_line_expression_file', type = str, default = '')
    parser.add_argument('--cell_line_gene_mapping_file', type = str, default = '')
    parser.add_argument('--cell_line_filter_mapping_file', type = str, default = '')
    parser.add_argument('--cell_line_filter_mapping_column', type = str, default = '')
    parser.add_argument('--cell_line_filter_inclusion_list', type = str, default = '')
    parser.add_argument('--cell_line_filter_exclusion_list', type = str, default = '')
    parser.add_argument('--cell_line_expression_mean_cut', type = float, default = -1.)
    parser.add_argument('--cell_line_expression_standardize_early', action = 'store_true')
    parser.add_argument('--cell_line_expression_filter_file', type = str, default = '')
    parser.add_argument('--no_gene_harmonization_union', action = 'store_true')
    parser.add_argument('--cell_line_drug_response_root_dir', type = str, default = '')
    parser.add_argument('--cell_line_drug_response_file', type = str, default = '')
    parser.add_argument('--cell_line_drug_response_row_info_file', type = str, default = '')
    parser.add_argument('--cell_line_drug_response_row_info_treat_col', type = str, default = '')
    parser.add_argument('--cell_line_drug_response_row_map_file', type = str, default = '')
    parser.add_argument('--cell_line_drug_response_target_column', type = str, default = '')
    parser.add_argument('--cell_line_drug_response_maxscale', action = 'store_true')
    parser.add_argument('--no_complete_data_shuffle', action = 'store_true')
    parser.add_argument('--complete_data_shuffle_seed', type = int, default = -1)
    parser.add_argument('--cache_data', action = 'store_true')
    parser.add_argument('--prefetch_data', action = 'store_true')
    
    # Debug
    parser.add_argument('--debug_data', action = 'store_true')
    parser.add_argument('--debug_training', action = 'store_true')
    parser.add_argument('--debug_gym', action = 'store_true')
    
    args = parser.parse_args()
    
    main(args)
