# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 2019

@author: rintala
"""

import tensorflow as tf
#import tensorflow_addons as tfa
import tensorflow_probability as tfp
import numpy as np
from copy import copy

class MODAE(tf.keras.Model):
    def __init__(
        self, 
        input_dim, 
        encoder_layers, 
        decoder_model = True, 
        decoder_layers = None, 
        reg_a = 1e-4, 
        fix_var = False, 
        variational = True, 
        variational_prior_logvar = 0.,
        input_noise = False, 
        noise_sd = 1.0,
        dropout_input = 0., 
        dropout_autoencoder = 0., 
        dropout_survival = 0., 
        dropout_classifier = 0., 
        dropout_batch = 0.,
        dropout_drug_response = 0.,
        reg_type = 'L2', 
        encoder_init_seeds = None, 
        decoder_init_seeds = None, 
        recon_init_seed = None, 
        supervised = False,
        classifier_layers = [], 
        classifier_init_seeds = None, 
        class_number = 2, 
        classifier_final_init_seed = None,
        regularize_reconstructive_layer = True,
        hidden_activation = 'relu', 
        hidden_initializer_function = tf.initializers.he_uniform, 
        hidden_dropout_function = tf.keras.layers.Dropout, 
        survival_model = False,
        survival_variational = False,
        survival_model_layers = [], 
        survival_model_init_seeds = None,
        survival_model_final_init_seed = None, 
        survival_covariate_n = 0,
        efron = True, 
        batch_adversarial_model = False,
        batch_adversarial_loss_function = 'wasserstein', 
        batch_adversarial_gradient_penalty = 10,  
        batch_adversarial_model_layers = [],
        batch_adversarial_model_init_seeds = None,
        batch_adversarial_model_pred_seed = None,
        batch_number = 2, 
        deconfounder_layers_per_batch = 0, 
        deconfounder_norm_penalty = 1., 
        deconfounder_centered_alignment = False, 
        drug_response_model = False,
        drug_response_model_output_activation = None,
        drug_response_model_layers = [],
        drug_response_model_drugwise_layers = [],
        drug_response_model_init_seeds = None,
        drug_response_model_drugwise_init_seeds = None, 
        drug_response_model_pred_seed = None,
        drug_number = None,
        objective_weights = [1.,1.,1.,1.,1.]
    ):
        super(MODAE, self).__init__()
        
        # Class constants
        self.batch_signs = {
            'cross-entropy' : tf.constant(1.), 
            'wasserstein' : tf.constant(-1.)}
        self.supervised_metric_signs = {
            'cross_entropy' : tf.constant(1.), 
            'auroc' : tf.constant(-1.), 
            'aupr' : tf.constant(-1.), 
            'acc' : tf.constant(-1.)}
        
        self.input_dim = copy(input_dim)
        self.encoder_layers = copy(encoder_layers)
        if len(encoder_layers) < 1:
            raise ValueError('Encoder should include at least 1 layer.')
        self.decoder_model = copy(decoder_model)
        if not decoder_layers is None:
            self.decoder_layers = copy(decoder_layers)
        else: 
            # Symmetric encoder-decoder
            self.decoder_layers = copy(encoder_layers)
            self.decoder_layers.reverse()
            # Drop last encoder layer (embedding layer / decoder input)
            self.decoder_layers[1:]
        self.reg_a = copy(reg_a)
        self.fix_var = copy(fix_var)
        self.variational = copy(variational)
        self.variational_prior_logvar = copy(variational_prior_logvar)
        self.input_noise = copy(input_noise)
        self.noise_sd = copy(input_noise)
        self.dropout_input = copy(dropout_input)
        self.dropout_autoencoder = copy(dropout_autoencoder)
        self.dropout_survival = copy(dropout_survival)
        self.dropout_classifier = copy(dropout_classifier)
        self.dropout_batch = copy(dropout_batch)
        self.dropout_drug_response = copy(dropout_drug_response)
        
        self.reg_type = copy(reg_type)
        self.encoder_init_seeds = copy(encoder_init_seeds)
        self.decoder_init_seeds = copy(decoder_init_seeds)
        self.recon_init_seed = copy(recon_init_seed)
        self.objective_weights = copy(objective_weights)
        
        # Supervised params
        self.supervised = copy(supervised)
        self.classifier_layers = copy(classifier_layers)
        self.classifier_init_seeds = copy(classifier_init_seeds)
        self.class_number = copy(class_number)
        self.classifier_final_init_seed = copy(classifier_final_init_seed)
        
        # Survival model params
        self.survival_model = copy(survival_model)
        self.efron = copy(efron)
        self.survival_variational = copy(survival_variational)
        self.survival_model_layers = copy(survival_model_layers)
        self.survival_model_init_seeds = copy(survival_model_init_seeds)
        self.survival_model_final_init_seed = copy(survival_model_final_init_seed)
        self.survival_covariate_n = copy(survival_covariate_n)
        
        # Adversarial model params
        self.batch_adversarial_model = copy(batch_adversarial_model)
        self.batch_adversarial_loss_function = copy(batch_adversarial_loss_function)
        self.batch_adversarial_loss_sign = self.batch_signs[batch_adversarial_loss_function]
        self.batch_adversarial_gradient_penalty = copy(batch_adversarial_gradient_penalty)
        self.batch_adversarial_model_layers = copy(batch_adversarial_model_layers)
        self.batch_adversarial_model_init_seeds = copy(batch_adversarial_model_init_seeds)
        self.batch_adversarial_model_pred_seed = copy(batch_adversarial_model_pred_seed)
        self.batch_number = copy(batch_number)
        
        # Deconfounder encoder params
        self.deconfounder_layers_per_batch = copy(deconfounder_layers_per_batch)
        self.deconfounder_model = self.deconfounder_layers_per_batch > 0
        self.deconfounder_norm_penalty = copy(deconfounder_norm_penalty)
        self.deconfounder_centered_alignment = copy(deconfounder_centered_alignment)
        
        # Drug response model params
        self.drug_response_model = copy(drug_response_model)
        self.drug_response_model_output_activation = copy(drug_response_model_output_activation)
        self.drug_response_model_layers = copy(drug_response_model_layers)
        self.drug_response_model_drugwise_layers = copy(drug_response_model_drugwise_layers)
        self.drug_response_model_init_seeds = copy(drug_response_model_init_seeds)
        self.drug_response_model_drugwise_init_seeds = copy(drug_response_model_drugwise_init_seeds)
        self.drug_response_model_pred_seed = copy(drug_response_model_pred_seed)
        self.drug_number = copy(drug_number)
        
        if reg_type == 'L2':
            regf1 = tf.keras.regularizers.l2(l = reg_a)
            regf2 = regf1
        elif reg_type == 'L1':
            regf1 = tf.keras.regularizers.l1(l = reg_a)
            regf2 = regf1
        elif reg_type == 'L(2,1)':
            regf1 = lambda x : tf.reduce_sum(tf.reduce_sum(tf.abs(x),0)**2.)**.5*reg_a
            regf2 = lambda x : regf1(tf.transpose(x))
        else:
            regf1 = None
            regf2 = None
        
        self.inference_net = tf.keras.Sequential()
        self.inference_net.add(tf.keras.layers.InputLayer(
            input_shape = (self.input_dim,)))
        if self.input_noise:
            self.inference_net.add(tf.keras.layers.GaussianNoise(self.noise_sd))
        if self.dropout_input > 0.0: 
            self.inference_net.add(tf.keras.layers.Dropout(self.dropout_input))
        
        for layer_i in range(len(self.encoder_layers) - 1):
            rseed = (None if self.encoder_init_seeds is None 
                     else self.encoder_init_seeds[layer_i])
            self.inference_net.add(tf.keras.layers.Dense(
                self.encoder_layers[layer_i], 
                activation = hidden_activation, 
                kernel_initializer = hidden_initializer_function(rseed),
                kernel_regularizer = regf1, use_bias = True))
            if self.dropout_autoencoder > 0.0: 
                self.inference_net.add(hidden_dropout_function(self.dropout_autoencoder))
        
        # Embedding layer
        layer_i = len(self.encoder_layers) - 1
        # For variational AE, the embedding layer is doubled to get both mean and logvar
        el_size_factor = 2 if self.variational else 1
        rseed = (None if self.encoder_init_seeds is None 
                 else self.encoder_init_seeds[layer_i])
        self.shared_layer_size = self.encoder_layers[layer_i]
        middle_layer_size = copy(self.shared_layer_size)
        if self.deconfounder_model:
            self.batch_specific_layer_mask = tf.concat(
                [
                    tf.ones((self.batch_number, middle_layer_size)), 
                    tf.repeat(tf.eye(self.batch_number), 
                        self.deconfounder_layers_per_batch, 
                        axis = 1)
                ], 
                axis = 1)
            middle_layer_size = self.batch_specific_layer_mask.shape[1]
        self.inference_net.add(tf.keras.layers.Dense(
            middle_layer_size * el_size_factor,
            kernel_initializer = tf.initializers.GlorotUniform(rseed),
            use_bias = True))
        
        self.middle_layer_size = middle_layer_size
        
        if self.decoder_model:
            self.generative_net = tf.keras.Sequential()
            self.generative_net.add(tf.keras.layers.InputLayer(input_shape = (self.middle_layer_size,)))
            
            for layer_i in range(len(self.decoder_layers)):
                rseed = (None if self.decoder_init_seeds is None 
                         else self.decoder_init_seeds[layer_i])
                self.generative_net.add(tf.keras.layers.Dense(
                    self.decoder_layers[layer_i], 
                    activation = hidden_activation, 
                    kernel_initializer = hidden_initializer_function(rseed),
                    kernel_regularizer = regf2, use_bias = True))
                if self.dropout_autoencoder > 0.0: 
                    self.generative_net.add(hidden_dropout_function(self.dropout_autoencoder))
            
            # Reconstructive layer
            if regularize_reconstructive_layer:
                reg_rec = regf2
            else:
                reg_rec = None
            rseed = self.recon_init_seed
            self.generative_net.add(tf.keras.layers.Dense(
                2 * self.input_dim if not self.fix_var else self.input_dim, # mean (and logvar)
                kernel_initializer = tf.initializers.GlorotUniform(rseed),
                kernel_regularizer = reg_rec, use_bias = True))
        
        # Supervised network
        if self.supervised:
            self.classifier_net = tf.keras.Sequential()
            self.classifier_net.add(tf.keras.layers.InputLayer(
                input_shape = (self.shared_layer_size,)))
            if self.dropout_classifier > 0.0: 
                self.classifier_net.add(tf.keras.layers.Dropout(self.dropout_classifier))
            for layer_i in range(len(self.classifier_layers)):
                rseed = (None if self.classifier_init_seeds is None 
                         else self.classifier_init_seeds[layer_i])
                self.classifier_net.add(tf.keras.layers.Dense(
                    self.classifier_layers[layer_i], 
                    activation = hidden_activation, 
                    kernel_initializer = hidden_initializer_function(rseed),
                    kernel_regularizer = regf1, use_bias = True))
                if self.dropout_classifier > 0.0: 
                    self.classifier_net.add(hidden_dropout_function(
                        self.dropout_classifier))
            rseed = self.classifier_final_init_seed
            self.classifier_net.add(tf.keras.layers.Dense(
                self.class_number, 
                kernel_initializer = tf.initializers.GlorotUniform(rseed),
                kernel_regularizer = regf1, 
                use_bias = True))
        
        # Survival model based on Cox PH
        # Covariates should be concatenated to input
        if self.survival_model:
            # Hazard components refer to final layer before Cox PH model
            # The last layer corresponds to covariates in Cox PH
            self.hazard_component_net = tf.keras.Sequential()
            self.hazard_component_net.add(tf.keras.layers.InputLayer(
                input_shape = (self.shared_layer_size + self.survival_covariate_n,)))
            if self.dropout_survival > 0.0: 
                self.hazard_component_net.add(tf.keras.layers.Dropout(self.dropout_survival))
            for layer_i in range(len(self.survival_model_layers)):
                rseed = (None if self.survival_model_init_seeds is None 
                         else self.survival_model_init_seeds[layer_i])
                self.hazard_component_net.add(tf.keras.layers.Dense(
                    self.survival_model_layers[layer_i], 
                    activation = hidden_activation, 
                    kernel_initializer = hidden_initializer_function(rseed),
                    kernel_regularizer = regf1, use_bias = True))
                if self.dropout_survival > 0.0: 
                    self.hazard_component_net.add(hidden_dropout_function(
                        self.dropout_survival))
            # Risk network combines risk components
            # For variational survival the final layer is doubled to get both mean and logvar
            sr_size_factor = 2 if self.survival_variational else 1
            self.survival_risk_net = tf.keras.Sequential()
            self.survival_risk_net.add(tf.keras.layers.InputLayer(
                input_shape = (self.survival_model_layers[layer_i],)))
            rseed = self.survival_model_final_init_seed
            self.survival_risk_net.add(tf.keras.layers.Dense(
                sr_size_factor, 
                kernel_initializer = tf.initializers.GlorotUniform(rseed),
                kernel_regularizer = regf1, 
                use_bias = True))
        
        # Batch adversarial model
        if self.batch_adversarial_model:
            self.batch_detector = tf.keras.Sequential()
            self.batch_detector.add(tf.keras.layers.InputLayer(input_shape = (self.shared_layer_size,)))
            if self.dropout_batch > 0.0:
                self.batch_detector.add(tf.keras.layers.Dropout(self.dropout_batch))
            for layer_i in range(len(self.batch_adversarial_model_layers)):
                rseed = (None if self.batch_adversarial_model_init_seeds is None 
                         else self.batch_adversarial_model_init_seeds[layer_i])
                self.batch_detector.add(tf.keras.layers.Dense(
                    self.batch_adversarial_model_layers[layer_i], 
                    activation = hidden_activation, 
                    kernel_initializer = hidden_initializer_function(rseed),
                    kernel_regularizer = regf1, use_bias = True))
                if self.dropout_batch > 0.0: 
                    self.batch_detector.add(hidden_dropout_function(self.dropout_batch))
            if batch_adversarial_loss_function == 'wasserstein':
                bd_out_size = 1
            else:
                bd_out_size = self.batch_number
            rseed = self.batch_adversarial_model_pred_seed
            self.batch_detector.add(tf.keras.layers.Dense(
                bd_out_size, 
                kernel_initializer = tf.initializers.GlorotUniform(rseed),
                kernel_regularizer = regf1, 
                use_bias = True))
            if (self.batch_adversarial_loss_function == 'wasserstein'
                and self.batch_adversarial_gradient_penalty == 0.):
                for w in self.batch_detector.trainable_variables:
                    w.assign(tf.clip_by_value(w, -.01, .01))
        
        # Drug response model
        if self.drug_response_model:
            self.drug_response_net = tf.keras.Sequential()
            self.drug_response_net.add(tf.keras.layers.InputLayer(input_shape = (self.shared_layer_size,)))
            if self.dropout_drug_response > 0.0:
                self.drug_response_net.add(tf.keras.layers.Dropout(
                    self.dropout_drug_response))
            last_shared_drug_layer_size = self.shared_layer_size
            for layer_i in range(len(self.drug_response_model_layers)):
                rseed = (None if self.drug_response_model_init_seeds is None 
                         else self.drug_response_model_init_seeds[layer_i])
                self.drug_response_net.add(tf.keras.layers.Dense(
                    self.drug_response_model_layers[layer_i], 
                    activation = hidden_activation, 
                    kernel_initializer = hidden_initializer_function(rseed),
                    kernel_regularizer = regf1, use_bias = True))
                if self.dropout_drug_response > 0.0: 
                    self.drug_response_net.add(hidden_dropout_function(
                        self.dropout_drug_response))
                last_shared_drug_layer_size = self.drug_response_model_layers[layer_i]
            if len(self.drug_response_model_drugwise_layers) == 0:
                rseed = self.drug_response_model_pred_seed
                self.drug_response_net.add(tf.keras.layers.Dense(
                    self.drug_number, 
                    activation = self.drug_response_model_output_activation, 
                    kernel_initializer = tf.initializers.GlorotUniform(rseed),
                    kernel_regularizer = regf1, 
                    use_bias = True))
                self.drugwise_model = False
            else:
                self.drugwise_model = True
                self.drug_response_drugwise_nets = []
                for drug_i in range(self.drug_number):
                    drnet_i = tf.keras.Sequential()
                    drnet_i.add(tf.keras.layers.InputLayer(
                        input_shape = (last_shared_drug_layer_size,)
                    ))
                    for layer_j in range(len(self.drug_response_model_drugwise_layers)):
                        rseed = (None if self.drug_response_model_drugwise_init_seeds is None 
                                 else self.drug_response_model_drugwise_init_seeds[layer_j])
                        drnet_i.add(tf.keras.layers.Dense(
                            self.drug_response_model_drugwise_layers[layer_j], 
                            activation = hidden_activation, 
                            kernel_initializer = hidden_initializer_function(rseed),
                            kernel_regularizer = regf1, use_bias = True))
                        if self.dropout_drug_response > 0.0: 
                            drnet_i.add(tf.keras.layers.Dropout(self.dropout_drug_response))
                    drnet_i.add(tf.keras.layers.Dense(
                        1, 
                        activation = self.drug_response_model_output_activation, 
                        kernel_initializer = tf.initializers.GlorotUniform(rseed),
                        kernel_regularizer = regf1, 
                        use_bias = True))
                    self.drug_response_drugwise_nets.append(drnet_i)
    
    @tf.function
    def sample(self, eps = None, N = 100):
        if eps is None:
            eps = tf.random.normal((N, self.middle_layer_size))
        return self.decode(eps) # TODO: modify eps if model has deconfounding
    
    @tf.function
    def encode(self, x, training = False):
        if self.variational:
            mean, logvar = tf.split(self.inference_net(x, training = training), 
                                    num_or_size_splits = 2, axis = 1)
            return mean, logvar
        else:
            mean = self.inference_net(x, training = training)
            return mean
    
    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape = mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean
    
    @tf.function
    def decode(self, z, b = None, training = False):
        if self.deconfounder_model:
            sample_z_mask = tf.gather(self.batch_specific_layer_mask, b, axis = 0)
            z = z * sample_z_mask
        if not self.fix_var:
            mean, logvar = tf.split(
                self.generative_net(z, training = training), 
                num_or_size_splits = 2, 
                axis = 1)
        else:
            mean = self.generative_net(z, training = training)
            logvar = mean * 0. # unit variance ~ MSE
        
        return mean, logvar
    
    @tf.function
    def call(self, x, b = None, training = False):
        if self.variational:
            z_mean, z_logvar = self.encode(x, training = training)
            recon_mean, recon_logvar = self.decode(z_mean, b = b, training = training)
            return recon_mean
        else:
            z_mean = self.encode(x, training = training)
            recon_mean, recon_logvar = self.decode(z_mean, b = b, training = training)
            return recon_mean
    
    @tf.function
    def batch_ce_loss(self, b, b_score, training = False):
        scce = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits = True,
            reduction = tf.keras.losses.Reduction.NONE)
        loss = self.batch_adversarial_loss_sign * scce(b, b_score)
        return tf.reduce_mean(loss)
    
    @tf.function
    def batch_correction_loss(self, z, b, training = False):
        b_score = self.batch_detector(z, training = training)
        if self.batch_adversarial_loss_function == 'wasserstein':
            b_label = tf.cast(b == 0, 'float32') * 2. - 1 # align data to batch 0
            loss = self.batch_adversarial_loss_sign *  b_score * b_label
            loss = tf.reduce_mean(tf.boolean_mask(loss, b != 0)) # exclude batch 0
        elif self.batch_adversarial_loss_function == 'cross-entropy':
            loss = self.batch_ce_loss(b, b_score, training = training)
        else:
            raise ValueError('Invalid batch adversarial loss function name: ' +
                             self.batch_adversarial_loss_function)
        return -loss # maximize
    
    @tf.function
    def batch_adversarial_loss(self, z, b, training = False):
        b_score = self.batch_detector(z, training = training)
        if self.batch_adversarial_loss_function == 'wasserstein':
            b_label = tf.cast(b == 0, 'float32') * 2. - 1 # align data to batch 0
            loss = self.batch_adversarial_loss_sign *  b_score * b_label
            loss = tf.reduce_mean(loss)
            if self.batch_adversarial_gradient_penalty > 0:
                loss += self.batch_detector_gradient_loss(z, b, training = training)
        elif self.batch_adversarial_loss_function == 'cross-entropy':
            loss = self.batch_ce_loss(b, b_score, training = training)
        else:
            raise ValueError('Invalid batch adversarial loss function name: ' +
                             self.batch_adversarial_loss_function)
        return loss
    
    @tf.function
    def batch_detector_gradient_loss(self, z, b, training = False):
        b_mask = tf.cast(b == 0, dtype = 'float32')
        d_ref = tfp.distributions.Categorical(probs = b_mask)
        d_alt = tfp.distributions.Categorical(probs = 1. - b_mask)
        z_ref = tf.gather(z, d_ref.sample(z.shape[0]), axis = 0)
        z_alt = tf.gather(z, d_alt.sample(z.shape[0]), axis = 0)
        eps = tf.random.uniform((z.shape[0], 1), 0., 1.)
        z_hat = eps * z_ref + (1. - eps) * z_alt
        with tf.GradientTape() as tape:
            tape.watch(z_hat)
            c_hat = self.batch_detector(z_hat, training = training)
        gradients = tape.gradient(c_hat, z_hat)
        term = tf.norm(gradients, axis = 1)
        reg = tf.reduce_mean(tf.square(term - 1.0))
        return reg * self.batch_adversarial_gradient_penalty
    
    @tf.function
    def confounder_loss(self, z, b):
        sample_z_mask = tf.gather(self.batch_specific_layer_mask, b, axis = 0)
        shared_embedding_mask = tf.expand_dims(tf.math.reduce_prod(self.batch_specific_layer_mask, axis = 0), axis = 0)
        private_embedding_mask = sample_z_mask * (1. - shared_embedding_mask)
        z_private = tf.boolean_mask(z * private_embedding_mask, shared_embedding_mask[0] == 0., axis = 1)
        z_shared = tf.boolean_mask(z * shared_embedding_mask, shared_embedding_mask[0] > 0., axis = 1)
        # Use centered alignment to induce orthogonality
        if self.deconfounder_centered_alignment:
            z_private_n = tf.math.reduce_sum(tf.boolean_mask(private_embedding_mask, shared_embedding_mask[0] == 0., axis = 1), axis = 0)
            z_private_mean = tf.math.reduce_sum(z_private, axis = 0) / z_private_n
            z_private_mean = tf.expand_dims(z_private_mean, axis = 0)
            z_shared_mean = tf.math.reduce_mean(z_shared, axis = 0)
            z_shared_mean = tf.expand_dims(z_shared_mean, axis = 0)
            m = tf.linalg.matmul(z_private - z_private_mean, z_shared - z_shared_mean, transpose_a = True)
        else:
            m = tf.linalg.matmul(z_private, z_shared, transpose_a = True)
        return (tf.math.reduce_sum(tf.square(m)) 
                / tf.cast(tf.math.reduce_prod(z.shape), dtype = 'float32'))
    
    @tf.function
    def drug_loss(self, z, dr, mask, training = False):
        n_screened = tf.reduce_sum(tf.cast(mask, 'float32'))
        if self.drugwise_model:
            sse = tf.constant(0., 'float32')
            dr_pred_pre = self.drug_response_net(z, training = training)
            for drug_i in range(self.drug_number):
                drug_model_i = self.drug_response_drugwise_nets[drug_i]
                dr_predi = drug_model_i(dr_pred_pre, training = training)
                dr_predi_nm = tf.ragged.boolean_mask(dr_predi, mask[:, drug_i])
                dri_nm = tf.ragged.boolean_mask(dr[:, drug_i], mask[:, drug_i])
                sei = (dri_nm - dr_predi_nm)**2 # Ragged array of squared losses
                sse += tf.reduce_sum(sei)
                #dr_pred.append(drug_model_i(dr_pred_pre, training = training))
            #dr_pred = tf.concat(dr_pred, axis = 1)
            return (sse, n_screened)
        else:
            dr_pred = self.drug_response_net(z, training = training)
            dr_pred_nm = tf.ragged.boolean_mask(dr_pred, mask)
            dr_nm = tf.ragged.boolean_mask(dr, mask)
            se = (dr_nm - dr_pred_nm)**2 # Ragged array of squared losses
            sse = tf.reduce_sum(se)
            return (sse, n_screened)
    
    @tf.function
    def compute_survival_loss_coxph(
            self, 
            z, 
            times, 
            events, 
            mask, 
            covariates = None, 
            training = False):
        z = tf.boolean_mask(z, mask, axis = 0)
        times = tf.boolean_mask(times, mask, axis = 0)
        events = tf.boolean_mask(events, mask, axis = 0)
        if not covariates is None:
            covariates = tf.boolean_mask(covariates, mask, axis = 0)

        if self.survival_covariate_n > 0:
            z_a = tf.concat([z, covariates], axis = 1)
        else:
            z_a = z
        hazard_components = self.hazard_component_net(z_a, training = training)
        log_hazard = self.survival_risk_net(hazard_components, training = training)
        
        # Each row indicates at risk group for sample i
        at_risk = tf.expand_dims(times, 0) >= tf.expand_dims(times, 1)
        log_hazard_at_risk = tf.ragged.boolean_mask(
            #tf.transpose(tf.broadcast_to(log_hazard, at_risk.shape)), 
            tf.transpose(log_hazard) * tf.ones_like(log_hazard), 
            at_risk)
        log_hazard_at_risk_max = tf.expand_dims(tf.math.reduce_max(log_hazard_at_risk, axis = 1), axis = 1)
        log_hazard_at_risk_max_events = tf.boolean_mask(log_hazard_at_risk_max, events)
        
        # At risk group total hazard for each sample
        #theta_at_risk = tf.math.exp(log_hazard_at_risk - tf.RaggedTensor.from_row_lengths(tf.repeat(log_hazard_at_risk_max, log_hazard_at_risk.row_lengths()), log_hazard_at_risk.row_lengths()))
        theta_at_risk = tf.math.exp(log_hazard_at_risk - log_hazard_at_risk_max) # identical to previous
        total_risk = tf.expand_dims(tf.math.reduce_sum(theta_at_risk, axis = 1), axis = 1)
        
        total_risk_events = tf.boolean_mask(total_risk, events)
        log_hazard_events = tf.boolean_mask(log_hazard, events)

        if self.efron:
            # For tied events (tj == ti)
            # Total risk and total tied risk are equal for all tied events.
            # And in practice we want to sum all of them, but tied events need to be
            # adjusted based on the number of tied events. Since total_tied_risk 
            # contains the |h| repeats of the sum, we can make a vector for the 
            # nominator and denominator for all events. Multiplying them element-
            # wise and summing we end up with the partial log likelihood.
            # To avoid numerical issues we will use calculate every log(sum(exp(x))) 
            # with a constant equal to the max x, i.e.: m + log(sum(exp(x-m))). 
            # The constant from above can be used since the largest element inside 
            # log is unchanged. 
            times_events = tf.boolean_mask(times, events)
            events_tied = tf.expand_dims(times_events, 1) == tf.expand_dims(times_events, 0)
            log_hazard_tied = tf.ragged.boolean_mask(
                #tf.transpose(tf.broadcast_to(log_hazard_events, events_tied.shape)), 
                tf.transpose(log_hazard_events) * tf.ones_like(log_hazard_events), 
                events_tied)
            theta_tied = tf.math.exp(log_hazard_tied - log_hazard_at_risk_max_events)
            total_tied_risk = tf.expand_dims(tf.math.reduce_sum(theta_tied, axis = 1), axis = 1)
            
            # The denominator is |h| which can be computed from tied_events matrix
            denominator = tf.reduce_sum(tf.cast(events_tied, dtype = 'float32'), axis = 0)
            # Nominator can be computed using the upper triangular part of the matrix - 1
            nominator = tf.reduce_sum(tf.linalg.band_part(tf.cast(events_tied, dtype = 'float32'), 0, -1), axis = 0) - 1
            
            term2 = (total_risk_events 
                     - total_tied_risk 
                     * tf.expand_dims(nominator / denominator, axis = 1))
            term2 = log_hazard_at_risk_max_events + tf.math.log(term2)
            
            log_likelihood = log_hazard_events - term2
        else:
            log_likelihood = (log_hazard_events 
                              - (log_hazard_at_risk_max_events 
                                 + tf.math.log(total_risk_events)))
        n_events = tf.reduce_sum(tf.cast(events, dtype = 'float32'))
        return log_likelihood, n_events
    
    @tf.function
    def survival_latent_distribution(
            self, 
            z, 
            times, 
            events, 
            mask, 
            covariates = None, 
            training = False):
        z = tf.boolean_mask(z, mask, axis = 0)
        times = tf.boolean_mask(times, mask, axis = 0)
        events = tf.boolean_mask(events, mask, axis = 0)
        if covariates is not None:
            covariates = tf.boolean_mask(covariates, mask, axis = 0)
        # Create input (x) from AE embedding + covariates
        if self.survival_covariate_n > 0:
            z_a = tf.concat([z, covariates], axis = 1)
        else:
            z_a = z
        hazard_components = self.hazard_component_net(z_a, training = training)
        st_mean, st_logvar = tf.split(
            self.survival_risk_net(hazard_components, training = training), 
            num_or_size_splits = 2, axis = 1)
        
        return st_mean, st_logvar
        
    @tf.function
    def compute_loss_survival_latent_model(
            self, 
            st_mean, 
            st_logvar, 
            times, 
            events, 
            mask, 
            covariates = None, 
            training = False):
        
        scales = tf.boolean_mask(tf.exp(.5 * st_logvar), tf.math.logical_not(events))
        upper_bound = tf.math.reduce_max(st_mean + 5. * scales)
        
        st_latent_dist = tfp.distributions.TruncatedNormal(
            loc = tf.boolean_mask(st_mean, tf.math.logical_not(events)), 
            scale = scales,
            low = tf.boolean_mask(times, tf.math.logical_not(events)),
            high = upper_bound)
        
        st_latent_times = st_latent_dist.sample()
        st_latent_logprob = st_latent_dist.log_prob(st_latent_times)
        
        st_observed_logprob = log_normal_pdf(
            tf.boolean_mask(times, events), 
            tf.boolean_mask(st_mean, events), 
            tf.boolean_mask(st_logvar, events))
        
        tf.reduce_mean(st_latent_logprob)
        tf.reduce_mean(st_observed_logprob)
        
        st_latent_prior = log_normal_pdf(st_latent_times, 0., 1.)
        st_observed_prior = log_normal_pdf(tf.boolean_mask(times, events), 0., 1.)

        st_latent_qz_x = log_normal_pdf(
            st_latent_times, 
            tf.boolean_mask(st_mean, tf.math.logical_not(events)), 
            tf.boolean_mask(st_logvar, tf.math.logical_not(events)))
        
        observed_logposterior = st_observed_logprob + st_observed_prior
        unobserved_elbo = st_latent_logprob + st_latent_prior - st_latent_qz_x
        
        return observed_logposterior + unobserved_elbo
    
    @tf.function
    def compute_loss_unsupervised(self, x, b = None, training = False, analytical = False):
        if self.variational:
            z_mean, z_logvar = self.encode(x, training = training)
            z = self.reparameterize(z_mean, z_logvar)
            recon_mean, recon_logvar = self.decode(z, b = b, training = training)
            
            if analytical:
                kl = .5 * tf.reduce_sum(1 + z_logvar - z_mean**2 
                                        - tf.math.exp(z_logvar), axis = 1)
                # Compute ELBO normalized to dimension size
                loss = (-1. * tf.reduce_mean(log_normal_pdf(x, recon_mean, recon_logvar), axis = 1) 
                        - kl / self.input_dim)
            else:
                # Assume diagonal covariance
                logpx_z = log_normal_pdf(x, recon_mean, recon_logvar)
                # TODO: Explore Poisson logpdf
                logpz = log_normal_pdf(z, 0., self.variational_prior_logvar)
                logqz_x = log_normal_pdf(z, z_mean, z_logvar)
                
                # Compute the dimension averaged 
                # - recon loss - embedding prior + embedding likelihood
                loss = (-1. * tf.reduce_mean(logpx_z, axis = 1) 
                        - (tf.reduce_sum(logpz, axis = 1) 
                           + tf.reduce_sum(logqz_x, axis = 1)) 
                        / self.input_dim)
        else:
            z = self.encode(x, training = training)
            recon_mean, recon_logvar = self.decode(z, b = b, training = training)
            mse = tf.reduce_mean((x - recon_mean)**2, axis = 1) # mean over features
            loss = mse
        
        return loss, z
    
    @tf.function#(input_signature=[])
    def compute_loss_supervised(self, z, y, mask, training = False):
        # For supervised loss only use non-missing labels
        y_logit = self.classifier_net(z, training = training)
        y_logit_nm = tf.boolean_mask(y_logit, mask, axis = 0)
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True,
                                                             reduction = tf.keras.losses.Reduction.NONE)
        loss = scce(tf.boolean_mask(y, mask, axis = 0), y_logit_nm)
        
        n_labelled = tf.reduce_sum(tf.cast(mask, 'float32'))
        # For compatibility with generic method for metrics we return all predictions
        return (loss, y_logit, n_labelled) 
    
    def set_trainability_with_checks(
        self, 
        autoencoder = False, 
        classifier = False,
        survival = False, 
        batch = False,
        drug_resp = False,
        set_all_true = False, 
        **kwargs
    ):
        if set_all_true:
            self.inference_net.trainable = True
            if self.decoder_model:
                self.generative_net.trainable = True
            if self.supervised:
                self.classifier_net.trainable = True
            if self.survival_model:
                self.hazard_component_net.trainable = True
                self.survival_risk_net.trainable = True
            if self.batch_adversarial_model:
                self.batch_detector.trainable = True
            if self.drug_response_model:
                self.drug_response_net.trainable = True
                if self.drugwise_model:
                    for drug_i in range(self.drug_number):
                        self.drug_response_drugwise_nets[drug_i].trainable = True
        else:
            self.inference_net.trainable = autoencoder
            if self.decoder_model:
                self.generative_net.trainable = autoencoder
            else:
                # No decoder means encoder must be always trainable
                self.inference_net.trainable = True
            if self.supervised:
                self.classifier_net.trainable = classifier
            if self.survival_model:
                self.hazard_component_net.trainable = survival
                self.survival_risk_net.trainable = survival
            if self.batch_adversarial_model:
                self.batch_detector.trainable = batch
            if self.drug_response_model:
                self.drug_response_net.trainable = drug_resp
                if self.drugwise_model:
                    for drug_i in range(self.drug_number):
                        self.drug_response_drugwise_nets[drug_i].trainable = drug_resp
    
    def train_step_gen(
        self, 
        optimizer, 
        ae = False, 
        classifier = False,
        survival = False, 
        batch_detector = False,
        batch_correction = False, 
        drug_resp = False, 
        patient_data = False, 
        cell_line_data = False, 
        return_z = False
    ):
        # Training step factory
        # tf.function will treat python variables as constants at compilation
        def train_step(model, **kwargs):
            if classifier and model.supervised:
                if kwargs.get('x', None) is None:
                    raise ValueError('Please define y when training supervised models.')
            with tf.GradientTape() as tape:
                loss = tf.constant(0., dtype = 'float32')
                if model.deconfounder_model:
                    shared_embedding_mask = tf.math.reduce_all(model.batch_specific_layer_mask > 0, axis = 0)
                z_combined = []
                if patient_data:
                    if ae:
                        patient_recon_loss, patient_z = model.compute_loss_unsupervised(
                            x = kwargs.get('x_patient'), 
                            b = kwargs.get('b_patient', None), 
                            training = True)
                        loss += tf.math.reduce_mean(patient_recon_loss) * model.objective_weights[0]
                        if batch_correction and model.deconfounder_model and model.deconfounder_norm_penalty > 0.:
                            loss += model.confounder_loss(patient_z, b = kwargs.get('b_patient')) * model.deconfounder_norm_penalty
                    else:
                        patient_z = model.encode(x = kwargs.get('x_patient'), training = True)
                        if model.variational:
                            patient_z = model.reparameterize(*patient_z)
                    if model.deconfounder_model:
                        patient_z = tf.boolean_mask(patient_z, shared_embedding_mask, axis = 1)
                    z_combined.append(patient_z)
                if cell_line_data:
                    if ae:
                        cl_recon_loss, cl_z = model.compute_loss_unsupervised(
                            x = kwargs.get('x_cl'), 
                            b = kwargs.get('b_cl', None), 
                            training = True)
                        loss += tf.math.reduce_mean(cl_recon_loss) * model.objective_weights[0]
                        if batch_correction and model.deconfounder_model and model.deconfounder_norm_penalty > 0.:
                            loss += model.confounder_loss(cl_z, b = kwargs.get('b_cl')) * model.deconfounder_norm_penalty
                    else:
                        cl_z = model.encode(x = kwargs.get('x_cl'), training = True)
                        if model.variational:
                            cl_z = model.reparameterize(*cl_z)
                    if model.deconfounder_model:
                        cl_z = tf.boolean_mask(cl_z, shared_embedding_mask, axis = 1)
                    z_combined.append(cl_z)
                if patient_data and cell_line_data and ae:
                    loss = loss / 2. # balanced average
                if (not patient_data) and (not cell_line_data):
                    if ae:
                        recon_loss, z = model.compute_loss_unsupervised(
                            kwargs.get('x'), 
                            kwargs.get('b', None), 
                            training = True)
                        loss += tf.math.reduce_mean(recon_loss) * model.objective_weights[0]
                        if batch_correction and model.deconfounder_model and model.deconfounder_norm_penalty > 0.:
                            loss += model.confounder_loss(z, b = kwargs.get('b')) * model.deconfounder_norm_penalty
                    else:
                        z = model.encode(x = kwargs.get('x'), training = True)
                        if model.variational:
                            z = model.reparameterize(*z)
                    if model.deconfounder_model:
                        z = tf.boolean_mask(z, shared_embedding_mask, axis = 1)
                    z_combined = [z]
                z_combined = tf.concat(z_combined, axis = 0)
                if classifier and model.supervised:
                    if patient_data:
                        z_for_class = patient_z
                    else:
                        z_for_class = z
                    prediction_loss, y_logit, n_labelled = model.compute_loss_supervised(
                        z = z_for_class, 
                        y = kwargs.get('y'), 
                        mask = kwargs.get('y_mask'), 
                        training = True)
                    loss += tf.reduce_sum(prediction_loss) / n_labelled * model.objective_weights[1]
                if survival and model.survival_model:
                    if patient_data:
                        z_for_surv = patient_z
                    else:
                        z_for_surv = z
                    surv_loglikelihood, n_events = model.compute_survival_loss_coxph(
                        z = z_for_surv, 
                        times = kwargs.get('survival_times'), 
                        events = kwargs.get('survival_events'), 
                        covariates = kwargs.get('survival_covariates'), 
                        mask = kwargs.get('survival_mask'), 
                        training = True)
                    loss -= tf.reduce_sum(surv_loglikelihood) / n_events * model.objective_weights[2] # maximize likelihood -> subtract
                if batch_detector and model.batch_adversarial_model:
                    b_loss = model.batch_adversarial_loss(
                        z_combined, 
                        kwargs.get('b'), 
                        training = True)
                    loss += b_loss * model.objective_weights[3]
                if batch_correction and model.batch_adversarial_model:
                    b_loss = model.batch_correction_loss(
                        z_combined, 
                        kwargs.get('b'), 
                        training = False) # No dropout for detector
                    loss += b_loss * model.objective_weights[3]
                if drug_resp and model.drug_response_model:
                    if cell_line_data:
                        z_for_dr = cl_z
                    else:
                        z_for_dr = z
                    dr_sse, n_screened = model.drug_loss(z = z_for_dr, 
                                                         dr = kwargs.get('dr'), 
                                                         mask = kwargs.get('dr_mask'), 
                                                         training = True)
                    loss += dr_sse / n_screened * model.objective_weights[4]
                loss += tf.math.add_n(model.losses) # regularization
            gradients = tape.gradient(loss, model.trainable_variables)
            nan_grads = tf.math.reduce_any([tf.math.reduce_any(tf.math.is_nan(i)) if not (i is None) else False for i in gradients])
            fin_grads = tf.math.reduce_all([tf.math.reduce_all(tf.math.is_finite(i)) if not (i is None) else False for i in gradients])
            if not nan_grads and fin_grads:
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                if (batch_detector 
                    and model.batch_adversarial_loss_function == 'wasserstein' 
                    and model.batch_adversarial_gradient_penalty == 0.):
                    for w in model.batch_detector.trainable_variables:
                        w.assign(tf.clip_by_value(w, -.01, .01))
            return z_combined
        train_step_tf = tf.function(train_step)
        # Control trainability
        def wrapped_train_step(
            model, 
            **kwargs
        ):
            model.set_trainability_with_checks(
                autoencoder = ae or batch_correction, 
                classifier = classifier,
                survival = survival, 
                batch = batch_detector,
                drug_resp = drug_resp)
            # Infer missing masks if necessary
            # TODO: consider alternatives
            if model.supervised and classifier and (kwargs.get('y_mask', None) is None):
                kwargs['y_mask'] = kwargs.get('y') >= 0
            if model.survival_model and survival and (kwargs.get('survival_mask', None) is None):
                kwargs['survival_mask'] = kwargs.get('survival_times') >= 0
            if model.batch_adversarial_model and (batch_detector or batch_correction) and (kwargs.get('b', None) is None):
                kwargs['b'] = None # TODO: implement
            if model.drug_response_model and drug_resp and (kwargs.get('dr_mask', None) is None):
                kwargs['dr_mask'] = kwargs.get('dr') >= 0
            z = train_step_tf(model = model, **kwargs)
            model.set_trainability_with_checks(set_all_true = True)
            if return_z:
                return z
            else:
                return
        
        return wrapped_train_step

def log_normal_pdf(sample, mean, logvar, raxis = 1):
    log2pi = tf.math.log(2. * np.pi)
    return -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)