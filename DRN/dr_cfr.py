import tensorflow as tf
import numpy as np

from DRN.util import *


class dr_cfr(object):
    """
    This file contains the class for DR-CFR.
    The network is implemented as a tensorflow graph.
    It is built upon implementation of the counterfactual regression neural network
    by F. Johansson, U. Shalit and D. Sontag: https://arxiv.org/abs/1606.03976
    """

    def __init__(self, x, t, y_, dims, do_in, do_out, p_t, r_alpha, r_lambda, r_beta, FLAGS, pi_0=None):
        self.variables = {}
        self.wd_loss = 0

        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.compat.v1.nn.elu
        else:
            self.nonlin = tf.compat.v1.nn.relu

        self._build_graph(x, t, y_, dims, do_in, do_out, p_t, r_alpha, r_lambda, r_beta, FLAGS, pi_0)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i)  # @TODO: not consistent with TF internally if changed
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = tf.compat.v1.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        self.wd_loss += wd * tf.compat.v1.nn.l2_loss(var)
        return var

    def _build_graph(self, x, t, y_, dims, do_in, do_out, p_t, p_alpha, p_lambda, p_beta, FLAGS, pi_0):
        """
        Constructs a TensorFlow subgraph for counterfactual regression.
        Sets the following member variables (to TF nodes):

        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.imb_loss       The imbalance term of the objective
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """

        r_alpha = p_alpha
        r_beta = p_beta
        r_lambda = p_lambda

        self.x = x
        self.t = t
        self.y_ = y_
        self.do_in = do_in
        self.do_out = do_out
        self.p_t = p_t
        self.r_alpha = r_alpha
        self.r_beta = r_beta
        self.r_lambda = r_lambda

        dim_input = dims[0]
        dim_in = dims[1]
        dim_out = dims[2]

        weights_in = [];
        biases_in = []

        if FLAGS.n_in == 0 or (FLAGS.n_in == 1 and FLAGS.varsel):
            dim_in = dim_input
        if FLAGS.n_out == 0:
            if FLAGS.split_output == False:
                dim_out = dim_in + 1
            else:
                dim_out = dim_in

        if FLAGS.batch_norm:
            bn_biases = []
            bn_scales = []

        ''' Construct input/representation layers '''
        h_rep_A, h_rep_norm_A, weights_in_A, biases_in_A = self._build_latent_graph(dim_input, dim_in, dim_out)
        h_rep_B, h_rep_norm_B, weights_in_B, biases_in_B = self._build_latent_graph(dim_input, dim_in, dim_out)
        h_rep_C, h_rep_norm_C, weights_in_C, biases_in_C = self._build_latent_graph(dim_input, dim_in, dim_out)
        weights_in = weights_in_A + weights_in_B + weights_in_C
        biases_in = biases_in_A + biases_in_B + biases_in_C

        ''' Construct ouput layers '''
        y, weights_out, weights_pred, biases_out, bias_pred = self._build_output_graph(
            tf.compat.v1.concat([h_rep_norm_B, h_rep_norm_C], 1), t, 2 * dim_in, dim_out, do_out, FLAGS)

        ''' Construct Pr( t | B ) '''
        W, b, pi0, cost = self._build_treatment_graph(h_rep_norm_B, dim_in)
        self.W = W
        self.b = b
        self.cost = cost
        if pi_0 == None:  # pi_0 not provided from file
            self.pi_0 = pi0
        else:
            self.pi_0 = pi_0

        if FLAGS.reweight_sample:
            w_t = t / (2. * p_t)
            w_c = (1. - t) / (2. * (1. - p_t))

            ''' Compute sample reweighting '''
            sample_weight = 1. * (1. + (1. - self.pi_0) / self.pi_0 * (p_t / (1. - p_t)) ** (2. * t - 1.)) * (w_t + w_c)

        else:
            sample_weight = 1.0

        # w_mean = tf.compat.v1.math.reduce_mean(sample_weight)
        # w_std = tf.compat.v1.math.reduce_std(sample_weight)
        # sample_weight = tf.compat.v1.clip_by_value(sample_weight, clip_value_min=w_mean-2.*w_std, clip_value_max=w_mean+2.*w_std)
        self.sample_weight = sample_weight

        ''' Construct Pr( t | (A,B) ) '''
        W_t, b_t, _, risk_t = self._build_treatment_graph(tf.compat.v1.concat([h_rep_norm_A, h_rep_norm_B], 1), 2 * dim_in)
        self.W_t = W_t
        self.b_t = b_t

        ''' Construct factual loss function '''
        if FLAGS.loss == 'l1':
            risk = tf.compat.v1.reduce_mean(sample_weight * tf.compat.v1.abs(y_ - y))
            pred_error = -tf.compat.v1.reduce_mean(res)
        elif FLAGS.loss == 'log':
            y = 0.995 / (1.0 + tf.compat.v1.exp(-y)) + 0.0025
            res = y_ * tf.compat.v1.log(y) + (1.0 - y_) * tf.compat.v1.log(1.0 - y)

            risk = -tf.compat.v1.reduce_mean(sample_weight * res)
            pred_error = -tf.compat.v1.reduce_mean(res)
        else:
            risk = tf.compat.v1.reduce_mean(sample_weight * tf.compat.v1.square(y_ - y))
            pred_error = tf.compat.v1.sqrt(tf.compat.v1.reduce_mean(tf.compat.v1.square(y_ - y)))

        ''' Regularization '''
        if FLAGS.p_lambda > 0 and FLAGS.rep_weight_decay:
            for i in range(0, FLAGS.n_in):
                if not (FLAGS.varsel and i == 0):  # No penalty on W in variable selection
                    self.wd_loss += tf.compat.v1.nn.l2_loss(weights_in[i])

        ''' Imbalance error '''
        imb_error, imb_dist = self._calculate_disc(h_rep_norm_C, r_alpha, FLAGS)

        ''' Total error '''
        tot_error = risk

        if FLAGS.p_alpha > 0:
            tot_error = tot_error + imb_error

        if FLAGS.p_beta > 0:
            tot_error = tot_error + r_beta * risk_t

        if FLAGS.p_lambda > 0:
            tot_error = tot_error + 1. * r_lambda * self.wd_loss

        if FLAGS.varsel:
            self.w_proj = tf.compat.v1.placeholder("float", shape=[dim_input], name='w_proj')
            self.projection = weights_in[0].assign(self.w_proj)

        self.output = y
        self.tot_loss = tot_error
        self.imb_loss = imb_error
        self.imb_dist = imb_dist
        self.pred_loss = pred_error

        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_pred = weights_pred
        self.biases_in = biases_in
        self.biases_out = biases_out
        self.bias_pred = bias_pred

        self.h_rep_norm = tf.compat.v1.concat([h_rep_norm_A,h_rep_norm_B,h_rep_norm_C],axis=1)

    def _build_latent_graph(self, dim_input, dim_in, dim_out):
        weights_in = [];
        biases_in = []

        h_in = [self.x]
        for i in range(0, FLAGS.n_in):
            if i == 0:
                ''' If using variable selection, first layer is just rescaling'''
                if FLAGS.varsel:
                    weights_in.append(tf.compat.v1.Variable(1.0 / dim_input * tf.compat.v1.ones([dim_input])))
                else:
                    weights_in.append(tf.compat.v1.Variable(
                        tf.compat.v1.compat.v1.random_normal([dim_input, dim_in], stddev=FLAGS.weight_init / np.sqrt(dim_input))))
            else:
                weights_in.append(
                    tf.compat.v1.Variable(tf.compat.v1.random_normal([dim_in, dim_in], stddev=FLAGS.weight_init / np.sqrt(dim_in))))

            ''' If using variable selection, first layer is just rescaling'''
            if FLAGS.varsel and i == 0:
                biases_in.append([])
                h_in.append(tf.compat.v1.mul(h_in[i], weights_in[i]))
            else:
                biases_in.append(tf.compat.v1.Variable(tf.compat.v1.zeros([1, dim_in])))
                z = tf.compat.v1.matmul(h_in[i], weights_in[i]) + biases_in[i]

                if FLAGS.batch_norm:
                    batch_mean, batch_var = tf.compat.v1.nn.moments(z, [0])

                    if FLAGS.normalization == 'bn_fixed':
                        z = tf.compat.v1.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                    else:
                        bn_biases.append(tf.compat.v1.Variable(tf.compat.v1.zeros([dim_in])))
                        bn_scales.append(tf.compat.v1.Variable(tf.compat.v1.ones([dim_in])))
                        z = tf.compat.v1.nn.batch_normalization(z, batch_mean, batch_var, bn_biases[-1], bn_scales[-1], 1e-3)

                h_in.append(self.nonlin(z))
                h_in[i + 1] = tf.compat.v1.nn.dropout(h_in[i + 1], self.do_in)

        h_rep = h_in[len(h_in) - 1]

        if FLAGS.normalization == 'divide':
            h_rep_norm = h_rep / safe_sqrt(tf.compat.v1.reduce_sum(tf.compat.v1.square(h_rep), axis=1, keep_dims=True))
        else:
            h_rep_norm = 1.0 * h_rep

        return h_rep, h_rep_norm, weights_in, biases_in

    def _build_output(self, h_input, dim_in, dim_out, do_out, FLAGS):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out] * FLAGS.n_out)

        weights_out = [];
        biases_out = []

        for i in range(0, FLAGS.n_out):
            wo = self._create_variable_with_weight_decay(
                tf.compat.v1.random_normal([dims[i], dims[i + 1]],
                                 stddev=FLAGS.weight_init / np.sqrt(dims[i])),
                'w_out_%d' % i, 1.0)
            weights_out.append(wo)

            biases_out.append(tf.compat.v1.Variable(tf.compat.v1.zeros([1, dim_out])))
            z = tf.compat.v1.matmul(h_out[i], weights_out[i]) + biases_out[i]
            # No batch norm on output because p_cf != p_f

            h_out.append(self.nonlin(z))
            h_out[i + 1] = tf.compat.v1.nn.dropout(h_out[i + 1], do_out)

        weights_pred = self._create_variable(tf.compat.v1.random_normal([dim_out, 1],
                                                              stddev=FLAGS.weight_init / np.sqrt(dim_out)), 'w_pred')
        bias_pred = self._create_variable(tf.compat.v1.zeros([1]), 'b_pred')

        if FLAGS.varsel or FLAGS.n_out == 0:
            self.wd_loss += tf.compat.v1.nn.l2_loss(
                tf.compat.v1.slice(weights_pred, [0, 0], [dim_out - 1, 1]))  # don't penalize treatment coefficient
        else:
            self.wd_loss += tf.compat.v1.nn.l2_loss(weights_pred)

        ''' Construct linear classifier '''
        h_pred = h_out[-1]
        y = tf.compat.v1.matmul(h_pred, weights_pred) + bias_pred

        return y, weights_out, weights_pred, biases_out, bias_pred

    def _build_output_graph(self, rep, t, dim_in, dim_out, do_out, FLAGS):
        ''' Construct output/regression layers '''

        if FLAGS.split_output:

            i0 = tf.compat.v1.to_int32(tf.compat.v1.where(t < 1)[:, 0])
            i1 = tf.compat.v1.to_int32(tf.compat.v1.where(t > 0)[:, 0])

            rep0 = tf.compat.v1.gather(rep, i0)
            rep1 = tf.compat.v1.gather(rep, i1)

            y0, weights_out0, weights_pred0, biases_out0, bias_pred0 = self._build_output(rep0, dim_in, dim_out, do_out,
                                                                                          FLAGS)
            y1, weights_out1, weights_pred1, biases_out1, bias_pred1 = self._build_output(rep1, dim_in, dim_out, do_out,
                                                                                          FLAGS)

            y = tf.compat.v1.dynamic_stitch([i0, i1], [y0, y1])
            weights_out = weights_out0 + weights_out1
            weights_pred = weights_pred0 + weights_pred1
            biases_out = biases_out0 + biases_out1
            bias_pred = bias_pred0 + bias_pred1
        else:
            h_input = tf.compat.v1.concat(1, [rep, t])
            y, weights_out, weights_pred, biases_out, bias_pred = self._build_output(h_input, dim_in + 1, dim_out,
                                                                                     do_out, FLAGS)

        return y, weights_out, weights_pred, biases_out, bias_pred

    def _calculate_disc(self, h_rep_norm, coef, FLAGS):
        t = self.t

        if FLAGS.use_p_correction:
            p_ipm = self.p_t
        else:
            p_ipm = 0.5

        if FLAGS.imb_fun == 'mmd2_rbf':
            imb_dist = mmd2_rbf(h_rep_norm, t, p_ipm, FLAGS.rbf_sigma)
            imb_error = coef * imb_dist
        elif FLAGS.imb_fun == 'mmd2_lin':
            imb_dist = mmd2_lin(h_rep_norm, t, p_ipm)
            imb_error = coef * mmd2_lin(h_rep_norm, t, p_ipm)
        elif FLAGS.imb_fun == 'mmd_rbf':
            imb_dist = tf.compat.v1.abs(mmd2_rbf(h_rep_norm, t, p_ipm, FLAGS.rbf_sigma))
            imb_error = safe_sqrt(tf.compat.v1.square(coef) * imb_dist)
        elif FLAGS.imb_fun == 'mmd_lin':
            imb_dist = mmd2_lin(h_rep_norm, t, p_ipm)
            imb_error = safe_sqrt(tf.compat.v1.square(coef) * imb_dist)
        elif FLAGS.imb_fun == 'wass':
            imb_dist, imb_mat = wasserstein(h_rep_norm, t, p_ipm, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations,
                                            sq=False, backpropT=FLAGS.wass_bpt)
            imb_error = coef * imb_dist
            self.imb_mat = imb_mat  # FOR DEBUG
        elif FLAGS.imb_fun == 'wass2':
            imb_dist, imb_mat = wasserstein(h_rep_norm, t, p_ipm, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations,
                                            sq=True, backpropT=FLAGS.wass_bpt)
            imb_error = coef * imb_dist
            self.imb_mat = imb_mat  # FOR DEBUG
        else:
            imb_dist = lindisc(h_rep_norm, t, p_ipm)
            imb_error = coef * imb_dist

        return imb_error, imb_dist

    def _build_treatment_graph(self, h_rep_norm, dim_in):
        t = self.t

        W = tf.compat.v1.Variable(tf.compat.v1.zeros([dim_in, 1]), name='W')
        b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='b')
        sigma = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(h_rep_norm, W) + b)
        pi_0 = tf.compat.v1.multiply(t, sigma) + tf.compat.v1.multiply(1.0 - t, 1.0 - sigma)
        cost = -tf.compat.v1.reduce_mean(
            tf.compat.v1.multiply(t, tf.compat.v1.log(sigma)) + tf.compat.v1.multiply(1.0 - t, tf.compat.v1.log(1.0 - sigma))) + 1e-3 * tf.compat.v1.nn.l2_loss(W)
        return W, b, pi_0, cost
