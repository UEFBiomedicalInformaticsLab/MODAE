# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 2020

@author: rintala
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import re
#import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from modae.evaluation_utilities import cross_validation_index

def complete_data_loader(
    patient_expression_root_dir = None, 
    patient_expression_cancer_list = [], 
    patient_expression_cancer_subset = [], # Subset after standardization
    patient_expression_file = None,
    patient_expression_sample_file = None, 
    patient_expression_sample_id_col  = 'colname', 
    patient_expression_sample_type_col = 'type',
    patient_expression_sample_type_include = ['1'],
    patient_gene_mapping_file = None, 
    patient_expression_log2 = True, 
    patient_expression_mean_cut = None, 
    patient_expression_transpose = True, 
    patient_expression_standardize_early = False, 
    patient_expression_filter_file = None, 
    patient_survival_file = None, 
    patient_survival_event_col = 'OS', 
    patient_survival_time_col = 'OS.time', 
    patient_survival_max_time = 3000, 
    patient_id_col = 'bcr_patient_barcode', 
    patient_id_tcga_barcode = True, 
    patient_redaction_col = 'Redaction', 
    patient_survival_covar_cols = [
        'gender', 
        'age_at_initial_pathologic_diagnosis', 
        'type'], 
    patient_survival_covar_onehot = [True, False, True], 
    cell_line_expression_root_dir = None, 
    cell_line_expression_file = None, 
    cell_line_gene_mapping_file = None, 
    cell_line_expression_log2 = False, 
    cell_line_expression_mean_cut = None, 
    cell_line_expression_transpose = False, 
    cell_line_expression_standardize_early = False, 
    cell_line_expression_filter_file = None, 
    cell_line_filter_mapping_file = None, 
    cell_line_filter_mapping_column = None, 
    cell_line_filter_inclusion_list = [], 
    cell_line_filter_exclusion_list = [], 
    gene_harmonization_union = False, 
    cell_line_drug_response_root_dir = None, 
    cell_line_drug_response_file = None, 
    cell_line_drug_response_row_info_file = None, 
    cell_line_drug_response_row_info_sample_col = 'sampleid', 
    cell_line_drug_response_row_info_treat_col = 'treatmentid_fixed', 
    cell_line_drug_response_row_map_file = None, 
    cell_line_drug_response_row_map_column = 'CCLE_model_id', 
    cell_line_drug_response_target_column = 'aac_recomputed', 
    cell_line_drug_response_maxscale = False, 
    shuffle = True, 
    shuffle_seed = None 
):
    cl_exp_flag = (
        (cell_line_expression_root_dir is not None) and 
        (cell_line_expression_file is not None))
    cl_dr_flag = (
        cl_exp_flag and 
        (cell_line_drug_response_root_dir is not None) and 
        (cell_line_drug_response_file is not None))
    p_exp_flag = (
        (patient_expression_root_dir is not None) and 
        (patient_expression_file is not None) and 
        (len(patient_expression_cancer_list)>0))
    p_surv_flag = (
        p_exp_flag and 
        (patient_survival_file is not None))
    
    if cl_exp_flag:
        fn = cell_line_expression_root_dir + cell_line_expression_file
        cl_data = pd.read_csv(fn, header = 0, index_col = 0)
        if cell_line_expression_transpose:
            cl_exp_id = cl_data.columns
            cl_gene_id = cl_data.index
        else:
            cl_exp_id = cl_data.index
            cl_gene_id = cl_data.columns
        cl_data_ind = cl_exp_id.to_numpy()
        cl_filter = None
        if len(cell_line_filter_inclusion_list) > 0:
            fn = cell_line_expression_root_dir + cell_line_filter_mapping_file
            cl_filter_table = pd.read_csv(fn, header = 0, index_col = 0)
            cl_filter_code = cl_filter_table.loc[
                cl_data_ind, 
                cell_line_filter_mapping_column]
            cl_filter = np.logical_not(cl_filter_code.isna().to_numpy())
            cl_filter[cl_filter] = np.isin(
                cl_filter_code[cl_filter].to_numpy(), 
                cell_line_filter_inclusion_list)
            if np.sum(cl_filter) == 0:
                raise ValueError('Cell-line filter returned 0 cell-lines.')
            cl_data_ind = cl_data_ind[cl_filter]
            if cell_line_expression_transpose:
                cl_data = cl_data.loc[:,cl_filter]
            else:
                cl_data = cl_data.loc[cl_filter,:]
        if len(cell_line_filter_exclusion_list) > 0:
            fn = cell_line_expression_root_dir + cell_line_filter_mapping_file
            cl_filter_table = pd.read_csv(fn, header = 0, index_col = 0)
            cl_filter_code = cl_filter_table.loc[
                cl_data_ind, 
                cell_line_filter_mapping_column]
            cl_filter = np.logical_not(cl_filter_code.isna().to_numpy())
            cl_filter[cl_filter] = np.logical_not(
                np.isin(cl_filter_code[cl_filter].to_numpy(), 
                        cell_line_filter_exclusion_list))
            if np.sum(cl_filter) == 0:
                raise ValueError('Cell-line filter returned 0 cell-lines.')
            cl_data_ind = cl_data_ind[cl_filter]
            if cell_line_expression_transpose:
                cl_data = cl_data.loc[:,cl_filter]
            else:
                cl_data = cl_data.loc[cl_filter,:]
        n_cl = cl_data_ind.shape[0]
    if cl_dr_flag:
        dr_fn = cell_line_drug_response_root_dir + cell_line_drug_response_file
        dr_data = pd.read_csv(dr_fn, header = 0, index_col = 0)
        drr_fn = cell_line_drug_response_root_dir + cell_line_drug_response_row_info_file
        dr_row_info = pd.read_csv(drr_fn, header = 0, index_col = 0)
        drclm_fn = cell_line_drug_response_root_dir + cell_line_drug_response_row_map_file
        dr_screen_cl_map = pd.read_csv(drclm_fn, header = 0, index_col = 0)
        
        nan_cl_ind = np.any(dr_row_info.isna().to_numpy(), axis = 1)
        dr_cl_id_mapped = dr_screen_cl_map.loc[
            dr_row_info[cell_line_drug_response_row_info_sample_col], 
            cell_line_drug_response_row_map_column]
        dr_cl_id_mapped_not_nan_ind = np.logical_not(dr_cl_id_mapped.isna().to_numpy())
        no_expr_cl = np.setdiff1d(
            dr_cl_id_mapped.to_numpy()[dr_cl_id_mapped_not_nan_ind], 
            cl_data_ind)
        no_expr_cl_ind = np.isin(dr_cl_id_mapped.to_numpy(), no_expr_cl)
        
        # exclude dr with mismatched cl ids
        dr_filter = np.logical_and(
            np.logical_not(nan_cl_ind), 
            dr_cl_id_mapped_not_nan_ind)
        # exclude dr without cl exp
        dr_filter = np.logical_and(
            dr_filter, 
            np.logical_not(no_expr_cl_ind))
        dr_missing_aac = dr_data[cell_line_drug_response_target_column].isna().to_numpy()
        # exclude dr with missing AAC
        dr_filter = np.logical_and(
            dr_filter, 
            np.logical_not(dr_missing_aac)) 
        dr_cl_vec = dr_cl_id_mapped[dr_filter].to_numpy(dtype = 'str')
        
        # Summarize mediums
        dr_treat_vec = dr_row_info.loc[
            dr_filter, 
            cell_line_drug_response_row_info_treat_col].to_numpy(dtype = 'str')
        dr_cl_unique, dr_cl_inverse = np.unique(dr_cl_vec, return_inverse = True)
        dr_treat_unique, dr_treat_inverse = np.unique(dr_treat_vec, return_inverse = True) 
        dr_table = np.full((dr_cl_unique.shape[0], dr_treat_unique.shape[0]), fill_value = 0.)
        dr_table_count = np.full(dr_table.shape, fill_value = 0.)
        dr_vec = dr_data[cell_line_drug_response_target_column].to_numpy()[dr_filter]
        if np.any(np.isnan(dr_vec)):
            raise ValueError('Unhandled missing values in drug response data.')
        
        # Compute average of drug response matrix
        for dr, t, cl in zip(dr_vec, dr_treat_inverse, dr_cl_inverse):
            dr_table[cl,t] += dr
            dr_table_count[cl,t] += 1.
        dr_table_count[dr_table_count == 0] = np.nan
        dr_table = dr_table / dr_table_count
        
        if cell_line_drug_response_maxscale:
            drug_max = np.nanmax(dr_table, axis = 0)
            dr_table = dr_table / np.expand_dims(drug_max, axis = 0)
        
        # Encode missing values as negative
        dr_table_missing = np.isnan(dr_table_count)
        dr_table[dr_table_missing] = -1
        dr_table_mask = np.logical_not(dr_table_missing)
        
        # Cell-line expression to dr index
        cl_name_map = dict(zip(dr_cl_unique, np.arange(dr_cl_unique.shape[0], dtype = 'int64')))
        cl_exp_ind = np.array([cl_name_map.get(i, -1) for i in cl_data_ind])
        
        if np.any(np.unique(cl_exp_ind[cl_exp_ind > -1], return_counts = True)[1] > 1):
            raise ValueError('Cell line expression ids match multiple drug response profiles.')
    
    if p_exp_flag:
        patient_expression_list = []
        for i in patient_expression_cancer_list:
            fni = f"{patient_expression_root_dir}{i}/{patient_expression_file}"
            pexi = pd.read_csv(fni, header = 0, index_col = 0) 
            if patient_expression_sample_file is not None:
                fnis = f"{patient_expression_root_dir}{i}/{patient_expression_sample_file}"
                pesi = pd.read_csv(fnis, header = 0, index_col = 0)
                #pesi = pesi.loc[pesi['view'] == 'mrna']
                pesi.set_index(patient_expression_sample_id_col, inplace = True)
                sample_id = [re.sub('\\.', '-', i) for i in pexi.columns]
                sample_type = pesi.loc[sample_id, patient_expression_sample_type_col]
                sample_ind = sample_type.astype('str').isin(patient_expression_sample_type_include)
                # Black magic to fix duplicated entries
                sample_ind = sample_ind.groupby(sample_ind.index).all()
                sample_ind = sample_ind.loc[sample_id]
                pexi = pexi.loc[:,sample_ind.to_numpy()]
            patient_expression_list.append(pexi)
        patient_expression = pd.concat(patient_expression_list, axis = 1)
        if patient_expression.isna().to_numpy().sum() > 0:
            raise ValueError('Missing values in patient expression profiles.')
        if patient_expression_transpose:
            patient_exp_id = patient_expression.columns
            patient_gene_id = patient_expression.index
        else:
            patient_exp_id = patient_expression.index
            patient_gene_id = patient_expression.columns
        if patient_id_tcga_barcode:
            patient_exp_id = np.array([re.sub('\\.', '-', i) for i in patient_exp_id])
        else:
            patient_exp_id = patient_exp_id.to_numpy()
        n_patient = patient_exp_id.shape[0]
    
    cl_exp_map_flag = False
    if cl_exp_flag:
        if cell_line_expression_transpose:
            cl_gene_id = cl_data.index
        else:
            cl_gene_id = cl_data.columns
        if cell_line_gene_mapping_file is not None:
            cl_exp_map_flag = True
            fn = cell_line_expression_root_dir + cell_line_gene_mapping_file
            cl_gene_map_df = pd.read_csv(fn, header = 0, index_col = 0)
            if np.any(cl_gene_id.to_numpy() != cl_gene_map_df['old_id'].to_numpy()):
                raise ValueError('Mismatch in gene id mapping of cell-line data.')
            cl_gene_id_harmonized = cl_gene_map_df['new_id']
        else:
            cl_gene_id_harmonized = cl_gene_id
        cl_gene_filter = np.logical_not(cl_gene_id_harmonized.isna().to_numpy())
        cl_gene_id_harmonized = cl_gene_id_harmonized.loc[cl_gene_filter].to_numpy()
    p_exp_map_flag = False
    if p_exp_flag:
        if patient_gene_mapping_file is not None:
            p_exp_map_flag = True
            fn = patient_expression_root_dir + patient_gene_mapping_file
            patient_gene_map_df = pd.read_csv(fn, header = 0, index_col = 0)
            if np.any(patient_gene_id.to_numpy() != patient_gene_map_df['old_id'].to_numpy()):
                raise ValueError('Mismatch in gene id mapping of patient data.')
            patient_gene_id_harmonized = patient_gene_map_df['new_id']
        else:
            patient_gene_id_harmonized = patient_gene_id
        patient_gene_filter = np.logical_not(patient_gene_id_harmonized.isna().to_numpy())
        patient_gene_id_harmonized = patient_gene_id_harmonized.loc[patient_gene_filter].to_numpy()
    
    preselection_gene_filter = []
    if patient_expression_filter_file is not None:
        fn = patient_expression_root_dir + patient_expression_filter_file
        with open(fn, 'r') as f:
            patient_gene_preselection = f.readlines()
        patient_gene_preselection = [
            re.sub('\n', '', i) for i in patient_gene_preselection]
        preselection_gene_filter = np.union1d(
            preselection_gene_filter, patient_gene_preselection)
    if cell_line_expression_filter_file is not None:
        fn = cell_line_drug_response_root_dir + cell_line_expression_filter_file
        with open(fn, 'r') as f:
            cl_gene_preselection = f.readlines()
        cl_gene_preselection = [re.sub('\n', '', i) for i in cl_gene_preselection]
        preselection_gene_filter = np.union1d(preselection_gene_filter, cl_gene_preselection)
    
    if cl_exp_flag and p_exp_flag:
        # Harmonize gene names (mapped or unmapped)
        if gene_harmonization_union:
            gene_id_harmonized_set = np.union1d(
                cl_gene_id_harmonized, 
                patient_gene_id_harmonized)
        else: # intersection
            gene_id_harmonized_set = np.intersect1d(cl_gene_id_harmonized, patient_gene_id_harmonized)
            
            cl_gene_in = np.isin(cl_gene_id_harmonized, gene_id_harmonized_set)
            patient_gene_in = np.isin(patient_gene_id_harmonized, gene_id_harmonized_set)
            
            cl_gene_filter[cl_gene_filter] = cl_gene_in
            patient_gene_filter[patient_gene_filter] = patient_gene_in
            
            cl_gene_id_harmonized = cl_gene_id_harmonized[cl_gene_in]
            patient_gene_id_harmonized = patient_gene_id_harmonized[patient_gene_in]
    else:
        if cl_exp_flag:
            gene_id_harmonized_set = cl_gene_id_harmonized
        if p_exp_flag:
            gene_id_harmonized_set = patient_gene_id_harmonized
    
    if cl_exp_map_flag or p_exp_map_flag:
        gene_id_map = dict(zip(
            gene_id_harmonized_set, 
            np.arange(gene_id_harmonized_set.shape[0], dtype = 'int64')))
    if cl_exp_map_flag:
        cl_gene_id_index = np.array([gene_id_map[i] for i in cl_gene_id_harmonized])
        cl_exp = accumulate_gene_expression(
            data = cl_data, 
            n = n_cl, 
            gene_id_harmonized_set = gene_id_harmonized_set, 
            gene_id_index = cl_gene_id_index, 
            gene_filter = cl_gene_filter, 
            transpose = cell_line_expression_transpose, 
            inverse_log2_transform = not cell_line_expression_log2)
    elif cl_exp_flag:
        cl_exp = cl_data.loc[cl_gene_filter, :].to_numpy()
    if cl_exp_flag and cell_line_expression_log2:
        cl_exp = np.log2(cl_exp + 1.)
    if cl_exp_flag:
        if cell_line_expression_mean_cut is not None:
            cl_gene_mean_filter = cl_exp.mean(axis = 0) > cell_line_expression_mean_cut
        else:
            cl_gene_mean_filter = np.full((cl_exp.shape[1],), True)
    
    if p_exp_map_flag:
        patient_gene_id_index = np.array([gene_id_map[i] for i in patient_gene_id_harmonized])
        patient_exp = accumulate_gene_expression(
            data = patient_expression, 
            n = n_patient, 
            gene_id_harmonized_set = gene_id_harmonized_set, 
            gene_id_index = patient_gene_id_index, 
            gene_filter = patient_gene_filter, 
            transpose = patient_expression_transpose, 
            inverse_log2_transform = not patient_expression_log2)
    elif p_exp_flag:
        patient_exp = patient_expression.loc[patient_gene_filter, :].to_numpy()
        if patient_expression_transpose:
            patient_exp = np.transpose(patient_exp)
    if p_exp_flag and patient_expression_log2:
        patient_exp = np.log2(patient_exp + 1.)
    if p_exp_flag:
        if patient_expression_mean_cut is not None:
            patient_gene_mean_filter = patient_exp.mean(axis = 0) > patient_expression_mean_cut
        else:
            patient_gene_mean_filter = np.full((patient_exp.shape[1],), True)
    
    if p_exp_flag and cl_exp_flag:
        gene_mean_filter = np.logical_and(
            patient_gene_mean_filter, 
            cl_gene_mean_filter)
        cl_exp = cl_exp[:,gene_mean_filter]
        patient_exp = patient_exp[:,gene_mean_filter]
        gene_id_harmonized_set = gene_id_harmonized_set[gene_mean_filter]
    elif cl_exp_flag:
        cl_exp = cl_exp[:,cl_gene_mean_filter]
        gene_id_harmonized_set = gene_id_harmonized_set[cl_gene_mean_filter]
    elif p_exp_flag:
        patient_exp = patient_exp[:,patient_gene_mean_filter]
        gene_id_harmonized_set = gene_id_harmonized_set[patient_gene_mean_filter]
        
    if p_exp_flag and patient_expression_standardize_early:
        patient_exp = StandardScaler().fit_transform(patient_exp)
    if cl_exp_flag and cell_line_expression_standardize_early:
        cl_exp = StandardScaler().fit_transform(cl_exp)
    
    if len(preselection_gene_filter) > 0:
        pre_gene_ind = np.isin(gene_id_harmonized_set, preselection_gene_filter)
        if p_exp_flag:
            patient_exp = patient_exp[:,pre_gene_ind]
        if cl_exp_flag:
            cl_exp = cl_exp[:,pre_gene_ind]
        gene_id_harmonized_set = gene_id_harmonized_set[pre_gene_ind]
    
    if p_exp_flag:
        cancer_type_label = [[name]*patient_expression_list[i].shape[1] for i, name in enumerate(patient_expression_cancer_list)]
        cancer_type_label = np.concatenate(cancer_type_label)
        if len(patient_expression_cancer_subset) > 0:
            patient_filter = np.isin(cancer_type_label, patient_expression_cancer_subset)
            patient_exp = patient_exp[patient_filter,:]
            cancer_type_label = cancer_type_label[patient_filter]
            patient_expression_cancer_list = patient_expression_cancer_subset
    else: 
        cancer_type_label = None
    
    if p_surv_flag:
        patient_survival_list = [
            pd.read_csv(
                patient_expression_root_dir + i + '/' + patient_survival_file, 
                header = 0, index_col = 0) for i in patient_expression_cancer_list]
        patient_survival = pd.concat(patient_survival_list, axis = 0)
        late_survival_ind = patient_survival[patient_survival_time_col] > patient_survival_max_time
        patient_survival.loc[late_survival_ind, patient_survival_time_col] = 3000.
        patient_survival.loc[late_survival_ind, patient_survival_event_col] = False
        patient_survival_ids = patient_survival[patient_id_col].to_numpy()
        # Survival time can be missing, need to filter
        patient_survival_filter = np.full((patient_survival.shape[0]), False)
        for i in np.concatenate([[patient_survival_event_col], 
                                 [patient_survival_time_col], 
                                 patient_survival_covar_cols]):
            temp = patient_survival[i].isna().to_numpy()
            patient_survival_filter = np.logical_or(patient_survival_filter, temp)
        patient_survival_filter = np.logical_not(patient_survival_filter)
        # Exclude redacted samples (consent, genetic mismatch)
        # A sample is redacted if Redaction is not NaN (in re-checked TCGA data)
        if patient_redaction_col.strip() != '':
            patient_survival_filter = np.logical_and(
                patient_survival_filter, 
                patient_survival[patient_redaction_col].isna().to_numpy())
        if patient_id_tcga_barcode:
            patient_exp_patient_id = np.array([re.sub('\\.', '-', i[:12]) for i in patient_expression.columns])
        else:
            patient_exp_patient_id = patient_expression.columns.to_numpy()
        patient_exp_patient_survival_filter =  np.isin(
            patient_exp_patient_id, 
            patient_survival_ids[patient_survival_filter])
        # Patient expression to survival index
        patient_exp_survival_map = dict(
            zip(
                patient_survival_ids[patient_survival_filter], 
                np.arange(patient_survival_filter.shape[0], dtype = 'int64')[patient_survival_filter]
            )
        )
        patient_exp_survival_ind = np.array([patient_exp_survival_map.get(i, -1) for i in patient_exp_patient_id])
        
        if np.any(np.unique(patient_exp_survival_ind[patient_exp_survival_ind > -1], return_counts = True)[1] > 1):
            # Same patient may have multiple biopsies, but only one survival record
            pass
    
    # Padding, separate patients and cell-lines
    # Survival is padded by using the last value (index = -1), ignored later via filter
    out = {}
    out['gene_ids'] = gene_id_harmonized_set
    if p_exp_flag:
        out['patient_exp'] = patient_exp # Rows are primary key for patients
        out['patient_rows'] = patient_exp_id
        if p_surv_flag:
            ps_time = patient_survival[patient_survival_time_col].iloc[patient_exp_survival_ind].to_numpy(dtype = 'float32')
            ps_event = patient_survival[patient_survival_event_col].iloc[patient_exp_survival_ind].to_numpy(dtype = 'bool')
            ps_covars = patient_survival.loc[:, patient_survival_covar_cols].iloc[patient_exp_survival_ind]
            ps_covars, ps_covar_cat = parse_covariates(ps_covars, patient_survival_covar_onehot)
            out['survival_time'] = ps_time
            out['survival_event'] = ps_event
            out['survival_covariates'] = ps_covars
            out['survival_covariates_categorical'] = ps_covar_cat
            out['survival_mask'] = patient_exp_patient_survival_filter
    if cl_exp_flag:
        out['cl_exp'] = cl_exp # Rows are primary key for CL
        out['cl_exp_rows'] = cl_data_ind
        if cl_dr_flag:
            cl_dr_table = dr_table[cl_exp_ind,:]
            cl_dr_table_mask = dr_table_mask[cl_exp_ind,:]
            cl_dr_table_mask[cl_exp_ind < 0, :] = False
            out['dr_table'] = cl_dr_table # Rows match primary key for CL
            out['dr_table_cols'] = dr_treat_unique
            out['dr_table_mask'] = cl_dr_table_mask
    
    if shuffle:
        rng = np.random.default_rng(seed = shuffle_seed)
        if p_exp_flag:
            patient_permutation = np.arange(out['patient_exp'].shape[0], dtype = 'int64')
            rng.shuffle(patient_permutation)
            out['patient_exp'] = out['patient_exp'][patient_permutation]
            out['patient_rows'] = out['patient_rows'][patient_permutation]
            if cancer_type_label is not None:
                out['patient_cancer_type'] = cancer_type_label
            if p_surv_flag:
                out['survival_time'] = out['survival_time'][patient_permutation]
                out['survival_event'] = out['survival_event'][patient_permutation]
                out['survival_covariates'] = out['survival_covariates'][patient_permutation]
                out['survival_covariates_categorical'] = out['survival_covariates_categorical']
                out['survival_mask'] = out['survival_mask'][patient_permutation]
        if cl_exp_flag:
            cl_permutation = np.arange(out['cl_exp'].shape[0], dtype = 'int64')
            rng.shuffle(cl_permutation)
            out['cl_exp'] = out['cl_exp'][cl_permutation]
            out['cl_exp_rows'] = out['cl_exp_rows'][cl_permutation]
            if cl_filter is not None:
                out['cl_cancer_type'] = cl_filter_code[cl_filter]
            if cl_dr_flag:
                out['dr_table'] = out['dr_table'][cl_permutation]
                out['dr_table_cols'] = out['dr_table_cols']
                out['dr_table_mask'] = out['dr_table_mask'][cl_permutation]
    
    return out

def accumulate_gene_expression(data, n, gene_id_harmonized_set, gene_id_index, gene_filter, transpose, inverse_log2_transform):
    out_exp = np.full((n, gene_id_harmonized_set.shape[0]), 0.)
    for i, j in zip(gene_id_index, np.argwhere(gene_filter)[:,0]):
        if transpose:
            expi = data.iloc[j,:].to_numpy()
        else:
            expi = data.iloc[:,j].to_numpy()
        if inverse_log2_transform:
            # Already log-transformed (assume pseudocount 1)
            out_exp[:,i] = out_exp[:,i] + np.power(2., expi) - 1.
        else:
            out_exp[:,i] += expi
    if inverse_log2_transform:
        out_exp = np.log2(out_exp + 1.)
    return out_exp

def parse_covariates(covariates, onehot):
    out = np.full((covariates.shape[0], 0), 1., dtype = 'float32')
    out_cat = np.full((0,), False)
    for cov, oh in zip(covariates.columns, onehot):
        if oh:
            coder = OneHotEncoder(sparse_output = False)
            cov_oh = coder.fit_transform(covariates[[cov]])
            out = np.concatenate([out, cov_oh.astype('float32')], axis = 1)
            out_cat = np.concatenate([out_cat, np.full((cov_oh.shape[1],), True)], axis = 0)
        else:
            out = np.concatenate([out, covariates[[cov]].to_numpy(dtype = 'float32')], axis = 1)
            out_cat = np.concatenate([out_cat, [False]], axis = 0)
    return out, out_cat

def process_patient_data(
    x, 
    cv_train_ind, 
    cv_test_ind = None, 
    supervised = False,
    survival_model = False,
    y = None, 
    y_mask = None, 
    survival_event = None,
    survival_time = None, 
    survival_covariates = None,
    survival_covariates_categorical = None,
    survival_mask = None,
    data_standardize = True, 
    std_scalers = {}
):
    out = {}
    model_spec = {}
    x_train = {}
    x_test = {}
    input_dim = {}
    for i in x.keys():
        x_train[i] = x[i][cv_train_ind,:]
        x_test[i] = x[i][cv_test_ind,:]
        if data_standardize:
            if std_scalers.get(i, None) is None:
                std_scalers[i] = StandardScaler().fit(x_train[i])
            x_train[i] = std_scalers[i].transform(x_train[i])
            if x_test[i].shape[0]:
                x_test[i] = std_scalers[i].transform(x_test[i])
        input_dim[i] =  x[i].shape[1]
    out['x_train'] = x_train
    out['x_test'] = x_test
    if data_standardize:
        out['x_scalers'] = std_scalers
    model_spec['input_dim'] = input_dim
    if supervised:
        out['y_train'] = y[cv_train_ind]
        out['y_test'] = y[cv_test_ind]
        if y_mask is None:
            y_mask = np.full(y.shape, True)
        out['y_mask_train'] = y_mask[cv_train_ind]
        out['y_mask_test'] = y_mask[cv_test_ind]
        y_u = np.unique(y[y_mask])
        y_n = y_u.shape[0]
        if np.any(y_u < 0):
            y_n -= 1
        model_spec['class_number'] = y_n
    
    if survival_model:
        if survival_mask is None:
            survival_mask = np.full(survival_event.shape[0], True)
        survival_std_scalers = {}
        out['survival_event_train'] = survival_event[cv_train_ind]
        out['survival_event_test'] = survival_event[cv_test_ind]
        out['survival_time_train'] = survival_time[cv_train_ind]
        out['survival_time_test'] = survival_time[cv_test_ind]
        out['survival_covariates_train'] = survival_covariates[cv_train_ind]
        out['survival_covariates_test'] = survival_covariates[cv_test_ind]
        out['survival_mask_train'] = survival_mask[cv_train_ind]
        out['survival_mask_test'] = survival_mask[cv_test_ind]
        for i in np.arange(survival_covariates.shape[1])[np.logical_not(survival_covariates_categorical)]:
            traini = out['survival_covariates_train'][:,i]
            survival_std_scalers[i] = StandardScaler().fit(np.expand_dims(traini[out['survival_mask_train']], axis = -1))
            traini[:] = survival_std_scalers[i].transform(np.expand_dims(traini, axis = -1))[:,0]
            traini[np.isnan(traini)] = 0. # will be masked out
            if out['survival_covariates_test'].shape[0] > 0:
                testi = out['survival_covariates_test'][:,i]
                testi[:] = survival_std_scalers[i].transform(np.expand_dims(testi, axis = -1))[:,0]
                testi[np.isnan(testi)] = 0. # will be masked out
        out['survival_std_scalers'] = survival_std_scalers
        model_spec['survival_covariate_n'] = survival_covariates.shape[1]
    return out, model_spec

def process_cl_data(
    x, 
    cv_train_ind, 
    cv_test_ind, 
    drug_response_model = False,
    drug_response = None,
    drug_response_mask = None, 
    data_standardize = True, 
    std_scalers = {}
):
    out = {}
    model_spec = {}
    x_train = {}
    x_test = {}
    input_dim = {}
    for i in x.keys():
        x_train[i] = x[i][cv_train_ind,:]
        x_test[i] = x[i][cv_test_ind,:]
        if data_standardize:
            if std_scalers.get(i, None) is None:
                std_scalers[i] = StandardScaler().fit(x_train[i])
            x_train[i] = std_scalers[i].transform(x_train[i])
            if x_test[i].shape[0]:
                x_test[i] = std_scalers[i].transform(x_test[i])
        input_dim[i] =  x[i].shape[1]
    out['x_train'] = x_train
    out['x_test'] = x_test
    if data_standardize:
        out['x_scalers'] = std_scalers
    model_spec['input_dim'] = input_dim
    if drug_response_model:
        out['drug_response_train'] = drug_response[cv_train_ind]
        out['drug_response_test'] = drug_response[cv_test_ind]
        if drug_response_mask is None: 
            # Assume full DR matrix without missing values ...
            drug_response_mask = np.full(drug_response.shape, True)
        out['drug_response_mask_train'] = drug_response_mask[cv_train_ind]
        out['drug_response_mask_test'] = drug_response_mask[cv_test_ind]
        model_spec['drug_number'] = drug_response.shape[1]
    return out, model_spec

# New data streaming system
# Serializes given samples to file
# None inputs will be ignored
def serialize_dataset_to_file(
    filename, 
    expression = None, 
    batch = None, 
    survival_time = None,
    survival_event = None,
    survival_covariates = None, 
    survival_mask = None,
    drug_response = None, 
    drug_response_mask = None,
    class_label = None,
    class_mask = None,
):
    input_size = {}
    input_dict = {}
    if not expression is None:
        input_dict['exp'] = expression.astype('float32')
        input_size['exp'] = expression.shape[0]
    if not expression is None:
        input_dict['batch'] = batch.astype('int64')
        input_size['batch'] = batch.shape[0]
    if not survival_time is None:
        input_dict['time'] = survival_time.astype('float32')
        input_size['time'] = survival_time.shape[0]
        if survival_mask is None:
            raise ValueError('Survival mask is required if serializing survival data.')
    if not survival_event is None:
        input_dict['event'] = survival_event.astype('bool')
        input_size['event'] = survival_event.shape[0]
        if survival_mask is None:
            raise ValueError('Survival mask is required if serializing survival data.')
    if not survival_covariates is None:
        input_dict['covar'] = survival_covariates.astype('float32')
        input_size['covar'] = survival_covariates.shape[0]
        if survival_mask is None:
            raise ValueError('Survival mask is required if serializing survival data.')
    if not survival_mask is None:
        input_dict['smask'] = survival_mask.astype('bool')
        input_size['smask'] = survival_mask.shape[0]
    if not drug_response is None:
        input_dict['dr'] = drug_response.astype('float32')
        input_size['dr'] = drug_response.shape[0]
        if drug_response_mask is None:
            raise ValueError('Drug response mask is required if serializing drug response data.')
    if not drug_response_mask is None:
        input_dict['drmask'] = drug_response_mask.astype('bool')
        input_size['drmask'] = drug_response_mask.shape[0]
    if not class_label is None:
        input_dict['class'] = class_label.astype('int32')
        input_size['class'] = class_label.shape[0]
        if class_mask is None:
            raise ValueError('Class mask is required if serializing class data.')
    if not class_mask is None:
        input_dict['cmask'] = class_mask.astype('bool')
        input_size['cmask'] = class_mask.shape[0]
    if np.unique(list(input_size.values())).shape[0] > 1:
        raise ValueError('Serialization data input sizes do no match.')
    input_ds = tf.data.Dataset.from_tensor_slices(input_dict)
    with tf.io.TFRecordWriter(filename) as fw:
        for i in input_ds:
            feature_dict = {}
            for j in input_dict.keys():
                feature_dict[j] = tf.train.Feature(bytes_list = tf.train.BytesList(value=[tf.io.serialize_tensor(i[j]).numpy()]))
            examplei = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            fw.write(examplei.SerializeToString())
    feature_spec = tf.io.FixedLenFeature([], dtype = tf.string)
    feature_spec_dict = dict(zip(input_dict.keys(), [feature_spec for j in input_dict.keys()]))
    return feature_spec_dict

RAW_TYPE = {
    'exp' : tf.float32,
    'batch' : tf.int64,
    'time' : tf.float32,
    'event' : tf.bool,
    'covar' : tf.float32,
    'smask' : tf.bool,
    'dr' : tf.float32,
    'drmask' : tf.bool,
    'class' : tf.int32,
    'cmask' : tf.bool}

RAGGED_MASK_KEY = {
    'time' : 'smask', 
    'event' : 'smask', 
    'covar' : 'smask', 
    'dr' : 'drmask', 
    'class' : 'cmask'}

def record_parser(record_bytes, feature_spec):
    out = tf.io.parse_single_example(record_bytes, features = feature_spec)
    for k in out.keys():
        out[k] = tf.io.parse_tensor(out[k], out_type = RAW_TYPE[k])
    return out

def parse_serialized_dataset_from_file(serialized_file, feature_spec):
    raw_dataset = tf.data.TFRecordDataset(serialized_file)
    processed_dataset = raw_dataset.map(
        lambda x : record_parser(x, feature_spec),
        num_parallel_calls=tf.data.AUTOTUNE)
    return processed_dataset

class CancerDataset:
    def __init__(self, 
                 x, 
                 y = None, 
                 time = None,
                 event = None,
                 covariates = None,
                 batch = None, 
                 drug_response = None):
        self.x = x
        if (not y is None) and (not x.shape[0] == y.shape[0]):
            raise ValueError('y input dimensions do not match x')
        self.y = y
        if (not time is None) and (not x.shape[0] == time.shape[0]):
            raise ValueError('time input dimensions do not match x')
        self.time = time
        if (not event is None) and (not x.shape[0] == event.shape[0]):
            raise ValueError('event input dimensions do not match x')
        self.event = event
        if (not covariates is None) and (not x.shape[0] == covariates.shape[0]):
            raise ValueError('covariates input dimensions do not match x')
        self.covariates = covariates
        if (not batch is None) and (not x.shape[0] == batch.shape[0]):
            raise ValueError('batch input dimensions do not match x')
        self.batch = batch
        if (not drug_response is None) and (not x.shape[0] == drug_response.shape[0]):
            raise ValueError('drug response input dimensions do not match x')
        self.drug_response = drug_response
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self,idx):
        x = self.x[idx,:]
        if self.y is None:
            y = tf.constant(-1, dtype = 'int64')
        else:
            y = self.y[idx]
        if self.time is None:
            time = tf.constant(-1, dtype = 'int64')
        else:
            time = self.time[idx]
        if self.event is None:
            event = tf.constant(False, dtype = 'bool')
        else:
            event = self.event[idx]
        if self.covariates is None:
            covariates = tf.constant([0.], dtype = 'float32')
        else:
            covariates = self.covariates[idx]
        if self.batch is None:
            batch = tf.constant(-1, dtype = 'int64')
        else:
            batch = self.batch[idx]
        if self.drug_response is None:
            drug_response = tf.constant([0.], dtype = 'float32')
        else:
            drug_response = self.drug_response[idx]
        return x, y, time, event, covariates, batch, drug_response
    
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
    
    def get_spec(self):
        if self.covariates is None:
            cv_out = tf.TensorSpec(shape = (1,), dtype = tf.float32)
        else:
            cv_out = tf.TensorSpec(shape = self.covariates.shape[1:], dtype = tf.float32)
        if self.drug_response is None:
            dr_out = tf.TensorSpec(shape = (1,), dtype = tf.float32)
        else:
            dr_out = tf.TensorSpec(shape = self.drug_response.shape[1:], dtype = tf.float32)
        out = (tf.TensorSpec(shape = self.x.shape[1:], dtype = tf.float32),
               tf.TensorSpec(shape = (), dtype = tf.int64),
               tf.TensorSpec(shape = (), dtype = tf.int64),
               tf.TensorSpec(shape = (), dtype = tf.bool),
               cv_out,
               tf.TensorSpec(shape = (), dtype = tf.int64),
               dr_out,)
        return out

def prepare_tf_dataset_from_numpy(x_train = None, 
                                  x_valid = None,
                                  y_train = None,
                                  y_valid = None,
                                  time_train = None,
                                  time_valid = None,
                                  event_train = None,
                                  event_valid = None,
                                  covariates_train = None,
                                  covariates_valid = None,
                                  b_train = None,
                                  b_valid = None,
                                  dr_train = None,
                                  dr_valid = None,
                                  valid_set = False):
    # Deal with possible missingness of arguments
    if x_valid is None or x_valid.shape[0] == 0:
        x_valid = x_train
    if y_valid is None or y_valid.shape[0] == 0:
        y_valid = y_train
    if event_valid is None or event_valid.shape[0] == 0:
        event_valid = event_train
    if time_valid is None or time_valid.shape[0] == 0:
        time_valid = time_train
    if covariates_valid is None or covariates_valid.shape[0] == 0:
        covariates_valid = covariates_train
    if b_valid is None or b_valid.shape[0] == 0:
        b_valid = b_train
    if dr_valid is None or dr_valid.shape[0] == 0:
        dr_valid = dr_train
    
    train_dataset_gen = CancerDataset(x_train, 
                                      y_train, 
                                      time_train, 
                                      event_train, 
                                      covariates_train,
                                      b_train,
                                      dr_train)
    train_dataset = tf.data.Dataset.from_generator(train_dataset_gen, 
                                                   output_signature = train_dataset_gen.get_spec())
    if valid_set:
        valid_dataset_gen = CancerDataset(x_valid, 
                                          y_valid, 
                                          time_valid, 
                                          event_valid, 
                                          covariates_valid,
                                          b_valid, 
                                          dr_valid)
        valid_dataset = tf.data.Dataset.from_generator(valid_dataset_gen, 
                                                       output_signature = valid_dataset_gen.get_spec())
    else:
        valid_dataset = None
    return train_dataset, valid_dataset

'''
Serialize training and test sets to files

Returns nested dictionary of serialized files and feature specifications
'''
def data_serialization(
    data_dict, # dictionary of input data from complete_data_loader
    nruns = 1, # CV repeats to generate from cv_seed
    nfolds = 5, # CV folds to generate from cv_seed 
    stratify_survival = True,
    stratify_subtype = True,
    stratify_dr = True, 
    stratify_time_quantiles = 10,
    cv_seed = 0, 
    model_args = {},
    data_standardize = False,
    file_name_prefix = '',
    omics_layer = 'mrna', 
    ps_validation_sets = True, 
    ps_test_sets = True
):
    patient_flag = np.isin('patient_exp', list(data_dict.keys()))
    #survival_flag = np.isin('survival_mask', list(data_dict.keys()))
    cl_flag = np.isin('cl_exp', list(data_dict.keys()))
    #cl_dr_flag = np.isin('dr_table', list(data_dict.keys()))
    
    N_patient = data_dict.get('patient_exp', np.array([])).shape[0]
    N_cl = data_dict.get('cl_exp', np.array([])).shape[0]
    
    stratify_patients = False
    strat_var = np.full((N_patient,),'')
    if model_args.get('supervised', False) and stratify_subtype:
        stratify_patients = True
        strat_var = np.char.add(strat_var, data_dict['y'].astype('str'))
    if model_args.get('survival_model', False) and stratify_survival:
        stratify_patients = True
        sevent = data_dict['survival_event']
        stimes = data_dict['survival_time']
        smask = data_dict['survival_mask']
        stq = np.quantile(stimes[smask], np.arange(1, stratify_time_quantiles) / stratify_time_quantiles)
        stbins = np.full(smask.shape, -1, dtype = 'int64')
        stbins[smask] = np.digitize(stimes[smask], stq)
        #stbins[stimes < 0] = -1 # Set all ambiguous/NaNs to -1 (own bin)
        strat_var = np.char.add(strat_var, sevent.astype('str'))
        strat_var = np.char.add(strat_var, stbins.astype('str'))
    if model_args.get('drug_response_model', False) and stratify_dr:
        cl_strat_var = np.any(data_dict['dr_table_mask'], axis = 1).astype('str')
    else:
        cl_strat_var = np.full((N_cl,),'')
    
    out = {}
    folds = np.arange(nfolds)
    runs = np.arange(nruns)
    for run in runs:
        out[str(run)] = {}
        if patient_flag:
            patient_cv_split = cross_validation_index(
                N = N_patient, 
                nfolds = nfolds,
                random_seed = cv_seed + run,
                labs = strat_var, 
                stratified = stratify_patients)
        if cl_flag:
            cl_cv_split = cross_validation_index(
                N = N_cl, 
                nfolds = nfolds,
                random_seed = cv_seed + run,
                labs = cl_strat_var, 
                stratified = stratify_dr)
        for fold in folds:
            out[str(run)][str(fold)] = {}
            # Add validation and test indices for training
            train_folds = []
            test_folds = []
            set_file_prefix = []
            if ps_validation_sets:
                # Evaluate parameters by training on k-2 folds and evaluating on 1 fold
                train_folds.append(np.setdiff1d(folds, np.array([fold, fold + 1]) % nfolds))
                test_folds.append(np.array([fold + 1]) % nfolds)
                set_file_prefix.append('ps_cv_')
            if ps_test_sets:
                # Evaluate parameters by training on k-1 folds and evaluating on 1 fold
                train_folds.append(np.setdiff1d(folds, np.array([fold]) % nfolds))
                test_folds.append(np.array([fold]) % nfolds)
                set_file_prefix.append('test_cv_')
            
            for cv_train_folds, cv_test_folds, file_string in zip(train_folds, test_folds, set_file_prefix):
                out[str(run)][str(fold)][file_string] = {}
                if patient_flag:
                    out[str(run)][str(fold)][file_string]['patient'] = {}
                    patient_train_ind = np.isin(patient_cv_split, cv_train_folds)
                    patient_test_ind = np.isin(patient_cv_split, cv_test_folds)
                    patient_data, patient_model_spec = process_patient_data(
                        x = {'mrna' : data_dict['patient_exp']}, 
                        cv_train_ind = patient_train_ind, 
                        cv_test_ind = patient_test_ind, 
                        supervised = model_args.get('supervised', False),
                        survival_model = model_args.get('survival_model', False),
                        y = data_dict.get('class', None), 
                        y_mask = data_dict.get('class_mask', None), 
                        survival_event = data_dict.get('survival_event', None),
                        survival_time = data_dict.get('survival_time', None), 
                        survival_covariates = data_dict.get('survival_covariates', None),
                        survival_covariates_categorical = data_dict.get('survival_covariates_categorical', None),
                        survival_mask = data_dict.get('survival_mask', None),
                        data_standardize = data_standardize)
                    patient_data['x_train'] = patient_data['x_train'][omics_layer]
                    patient_data['x_test'] = patient_data['x_test'][omics_layer]
                    patient_data['b_train'] = np.full((patient_data['x_train'].shape[0],), 0, dtype = 'int64')
                    patient_data['b_test'] = np.full((patient_data['x_test'].shape[0],), 0, dtype = 'int64')
                    patient_model_spec['input_dim'] = patient_model_spec['input_dim'][omics_layer]
                    patient_info = {}
                    patient_info['rownames_train'] = data_dict['patient_rows'][patient_train_ind]
                    patient_info['rownames_test'] = data_dict['patient_rows'][patient_test_ind]
                    
                    for cv_set in ['train', 'test']:
                        fn = file_name_prefix + file_string + 'run' + str(run) + '_fold' + str(fold) + '_patient' + '_' + cv_set + '.tfrecords'
                        patient_feature_spec = serialize_dataset_to_file(
                            filename = fn,
                            expression = patient_data['x_' + cv_set], 
                            batch = patient_data['b_' + cv_set], 
                            survival_time = patient_data.get('survival_time_' + cv_set, None),
                            survival_event = patient_data.get('survival_event_' + cv_set, None),
                            survival_covariates = patient_data.get('survival_covariates_' + cv_set, None),
                            survival_mask = patient_data.get('survival_mask_' + cv_set, None),
                            class_label = patient_data.get('y_' + cv_set, None),
                            class_mask = patient_data.get('y_mask_' + cv_set, None))
                        out[str(run)][str(fold)][file_string]['patient'][cv_set] = {
                            'filename' : fn,
                            'feature_spec' : patient_feature_spec,
                            'model_spec' : patient_model_spec,
                            'sample_info' : {'rownames' : patient_info['rownames_' + cv_set]}}
                if cl_flag:
                    out[str(run)][str(fold)][file_string]['cell_line'] = {}
                    cl_train_ind = np.isin(cl_cv_split, cv_train_folds)
                    cl_test_ind = np.isin(cl_cv_split, cv_test_folds)
                    cl_data, cl_model_spec = process_cl_data(
                        x = {'mrna' : data_dict['cl_exp']}, 
                        cv_train_ind = cl_train_ind, 
                        cv_test_ind = cl_test_ind, 
                        drug_response_model = model_args.get('drug_response_model', False),
                        drug_response = data_dict.get('dr_table', None), 
                        drug_response_mask = data_dict.get('dr_table_mask', None), 
                        data_standardize = data_standardize)
                    
                    cl_data['x_train'] = cl_data['x_train'][omics_layer]
                    cl_data['x_test'] = cl_data['x_test'][omics_layer]
                    cl_data['b_train'] = np.full((cl_data['x_train'].shape[0],), 1, dtype = 'int64')
                    cl_data['b_test'] = np.full((cl_data['x_test'].shape[0],), 1, dtype = 'int64')
                    cl_model_spec['input_dim'] = cl_model_spec['input_dim'][omics_layer]
                    cl_info = {}
                    cl_info['rownames_train'] = data_dict['cl_exp_rows'][cl_train_ind]
                    cl_info['rownames_test'] = data_dict['cl_exp_rows'][cl_test_ind]
                    
                    for cv_set in ['train', 'test']:
                        fn = file_name_prefix + file_string + 'run' + str(run) + '_fold' + str(fold) + '_cl' + '_' + cv_set + '.tfrecords'
                        cl_feature_spec = serialize_dataset_to_file(
                            filename = fn, 
                            expression = cl_data['x_' + cv_set], 
                            batch = cl_data['b_' + cv_set], 
                            drug_response = cl_data.get('drug_response_' + cv_set, None), 
                            drug_response_mask = cl_data.get('drug_response_mask_' + cv_set, None))
                        out[str(run)][str(fold)][file_string]['cell_line'][cv_set] = {
                            'filename' : fn,
                            'feature_spec' : cl_feature_spec,
                            'model_spec' : cl_model_spec, 
                            'sample_info' : {'rownames' : cl_info['rownames_' + cv_set]}}
    return out

def dataset_batch_setup(
    dataset,
    cache = False,
    shuffle = True,
    repeat = False, 
    mini_batches = True,
    batch_size = 256,
    prefetch = True,
    prefetch_size = tf.data.AUTOTUNE
):
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(len(dataset), seed = tf.constant(1, tf.int64))
    if repeat:
        dataset = dataset.repeat(count = -1)
    if mini_batches:
        dataset = dataset.batch(batch_size)
    if prefetch:
        dataset = dataset.prefetch(prefetch_size)
    return dataset


def FeatureSpecEncoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tf.io.FixedLenFeature):
        return 'tf.io.FixedLenFeature'
    if isinstance(obj, dict):
        return dict([(k, FeatureSpecEncoder(i)) for k,i in obj.items()])
    if isinstance(obj, list):
        return [FeatureSpecEncoder(i) for i in obj]
    if isinstance(obj, tuple):
        return tuple([FeatureSpecEncoder(i) for i in obj])
    return obj

def JSONFeatureSpecDecoder(obj):
    if isinstance(obj, dict):
        return dict([(k, JSONFeatureSpecDecoder(i)) for k,i in obj.items()])
    if isinstance(obj, list):
        return [JSONFeatureSpecDecoder(i) for i in obj]
    if isinstance(obj, tuple):
        return tuple([JSONFeatureSpecDecoder(i) for i in obj])
    if isinstance(obj, str):
        if obj == 'tf.io.FixedLenFeature':
            return tf.io.FixedLenFeature([], dtype = tf.string)
    return obj