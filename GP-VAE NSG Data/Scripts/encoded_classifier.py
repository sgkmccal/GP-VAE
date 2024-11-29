import sys
import os
import time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
tfd = tfp.distributions


#%% NN utils
def make_nn(output_size, hidden_sizes):
    """ Creates fully connected neural network
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
    """
    layers = [tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32)
              for h in hidden_sizes]
    layers.append(tf.keras.layers.Dense(output_size, dtype=tf.float32))
    return tf.keras.Sequential(layers)


def make_cnn(output_size, hidden_sizes, kernel_size=3):
    """ Construct neural network consisting of
          one 1d-convolutional layer that utilizes temporal dependences,
          fully connected network

        :param output_size: output dimensionality
        :param hidden_sizes: tuple of hidden layer sizes.
                             The tuple length sets the number of hidden layers.
        :param kernel_size: kernel size for convolutional layer
    """
    cnn_layer = [tf.keras.layers.Conv1D(hidden_sizes[0], kernel_size=kernel_size,
                                        padding="same", dtype=tf.float32)]
    layers = [tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32)
              for h in hidden_sizes[1:]]
    layers.append(tf.keras.layers.Dense(output_size, dtype=tf.float32))
    return tf.keras.Sequential(cnn_layer + layers)


def make_2d_cnn(output_size, hidden_sizes, kernel_size=3):
    """ Creates fully convolutional neural network.
        Used as CNN preprocessor for image data (HMNIST, SPRITES)

        :param output_size: output dimensionality
        :param hidden_sizes: tuple of hidden layer sizes.
                             The tuple length sets the number of hidden layers.
        :param kernel_size: kernel size for convolutional layers
    """
    layers = [tf.keras.layers.Conv2D(h, kernel_size=kernel_size, padding="same",
                                     activation=tf.nn.relu, dtype=tf.float32)
              for h in hidden_sizes + [output_size]]
    return tf.keras.Sequential(layers)

#%%''' TF utils '''
def reduce_logmeanexp(x, axis, eps=1e-5):
    """Numerically-stable (?) implementation of log-mean-exp.
    Args:
        x: The tensor to reduce. Should have numeric type.
        axis: The dimensions to reduce. If `None` (the default),
              reduces all dimensions. Must be in the range
              `[-rank(input_tensor), rank(input_tensor)]`.
        eps: Floating point scalar to avoid log-underflow.
    Returns:
        log_mean_exp: A `Tensor` representing `log(Avg{exp(x): x})`.
    """
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    return tf.log(tf.reduce_mean(
            tf.exp(x - x_max), axis=axis, keepdims=True) + eps) + x_max

def multiply_tfd_gaussians(gaussians):
    """Multiplies two tfd.MultivariateNormal distributions."""
    mus = [gauss.mean() for gauss in gaussians]
    Sigmas = [gauss.covariance() for gauss in gaussians]
    mu_3, Sigma_3, _ = multiply_gaussians(mus, Sigmas)
    return tfd.MultivariateNormalFullCovariance(loc=mu_3, covariance_matrix=Sigma_3)

def multiply_inv_gaussians(mus, lambdas):
    """Multiplies a series of Gaussians that is given as a list of mean vectors and a list of precision matrices.
    mus: list of mean with shape [n, d]
    lambdas: list of precision matrices with shape [n, d, d]
    Returns the mean vector, covariance matrix, and precision matrix of the product
    """
    assert len(mus) == len(lambdas)
    batch_size = int(mus[0].shape[0])
    d_z = int(lambdas[0].shape[-1])
    identity_matrix = tf.reshape(tf.tile(tf.eye(d_z), [batch_size,1]), [-1,d_z,d_z])
    lambda_new = tf.reduce_sum(lambdas, axis=0) + identity_matrix
    mus_summed = tf.reduce_sum([tf.einsum("bij, bj -> bi", lamb, mu)
                                for lamb, mu in zip(lambdas, mus)], axis=0)
    sigma_new = tf.linalg.inv(lambda_new)
    mu_new = tf.einsum("bij, bj -> bi", sigma_new, mus_summed)
    return mu_new, sigma_new, lambda_new

def multiply_inv_gaussians_batch(mus, lambdas):
    """Multiplies a series of Gaussians that is given as a list of mean vectors and a list of precision matrices.
    mus: list of mean with shape [..., d]
    lambdas: list of precision matrices with shape [..., d, d]
    Returns the mean vector, covariance matrix, and precision matrix of the product
    """
    assert len(mus) == len(lambdas)
    batch_size = mus[0].shape.as_list()[:-1]
    d_z = lambdas[0].shape.as_list()[-1]
    identity_matrix = tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(d_z), axis=0), axis=0), batch_size+[1,1])
    lambda_new = tf.reduce_sum(lambdas, axis=0) + identity_matrix
    mus_summed = tf.reduce_sum([tf.einsum("bcij, bcj -> bci", lamb, mu)
                                for lamb, mu in zip(lambdas, mus)], axis=0)
    sigma_new = tf.linalg.inv(lambda_new)
    mu_new = tf.einsum("bcij, bcj -> bci", sigma_new, mus_summed)
    return mu_new, sigma_new, lambda_new

def multiply_gaussians(mus, sigmas):
    """Multiplies a series of Gaussians that is given as a list of mean vectors and a list of covariance matrices.
    mus: list of mean with shape [n, d]
    sigmas: list of covariance matrices with shape [n, d, d]
    Returns the mean vector, covariance matrix, and precision matrix of the product
    """
    assert len(mus) == len(sigmas)
    batch_size = [int(n) for n in mus[0].shape[0]]
    d_z = int(sigmas[0].shape[-1])
    identity_matrix = tf.reshape(tf.tile(tf.eye(d_z), [batch_size,1]), batch_size+[d_z,d_z])
    sigma_new = identity_matrix
    mu_new = tf.zeros((batch_size, d_z))
    for mu, sigma in zip(mus, sigmas):
        sigma_inv = tf.linalg.inv(sigma_new + sigma)
        sigma_prod = tf.matmul(tf.matmul(sigma_new, sigma_inv), sigma)
        mu_prod = (tf.einsum("bij,bj->bi", tf.matmul(sigma, sigma_inv), mu_new)
                   + tf.einsum("bij,bj->bi", tf.matmul(sigma_new, sigma_inv), mu))
        sigma_new = sigma_prod
        mu_new = mu_prod
    lambda_new = tf.linalg.inv(sigma_new)
    return mu_new, sigma_new, lambda_new

#%% Encoder functions
class DiagonalEncoder(tf.keras.Model):
    def __init__(self, z_size, hidden_sizes=(64, 64), **kwargs):
        """ Encoder with factorized Normal posterior over temporal dimension
            Used by disjoint VAE and HI-VAE with Standard Normal prior
            :param z_size: latent space dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
        """
        super(DiagonalEncoder, self).__init__()
        self.z_size = int(z_size)
        self.net = make_nn(2*z_size, hidden_sizes)

    def __call__(self, x):
        mapped = self.net(x)
        return tfd.MultivariateNormalDiag(
          loc=mapped[..., :self.z_size],
          scale_diag=tf.nn.softplus(mapped[..., self.z_size:]))

class JointEncoder(tf.keras.Model):
    def __init__(self, z_size, hidden_sizes=(64, 64), window_size=3, transpose=False, **kwargs):
        """ Encoder with 1d-convolutional network and factorized Normal posterior
            Used by joint VAE and HI-VAE with Standard Normal prior or GP-VAE with factorized Normal posterior
            :param z_size: latent space dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
            :param window_size: kernel size for Conv1D layer
            :param transpose: True for GP prior | False for Standard Normal prior
        """
        super(JointEncoder, self).__init__()
        self.z_size = int(z_size)
        self.net = make_cnn(2*z_size, hidden_sizes, window_size)
        self.transpose = transpose

    def __call__(self, x):
        mapped = self.net(x)
        if self.transpose:
            num_dim = len(x.shape.as_list())
            perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
            mapped = tf.transpose(mapped, perm=perm)
            return tfd.MultivariateNormalDiag(
                    loc=mapped[..., :self.z_size, :],
                    scale_diag=tf.nn.softplus(mapped[..., self.z_size:, :]))
        return tfd.MultivariateNormalDiag(
                    loc=mapped[..., :self.z_size],
                    scale_diag=tf.nn.softplus(mapped[..., self.z_size:]))

class BandedJointEncoder(tf.keras.Model):
    def __init__(self, z_size, hidden_sizes=(64, 64), window_size=3, data_type=None, **kwargs):
        """ Encoder with 1d-convolutional network and multivariate Normal posterior
            Used by GP-VAE with proposed banded covariance matrix
            :param z_size: latent space dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
            :param window_size: kernel size for Conv1D layer
            :param data_type: needed for some data specific modifications, e.g:
                tf.nn.softplus is a more common and correct choice, however
                tf.nn.sigmoid provides more stable performance on Physionet dataset
        """
        super(BandedJointEncoder, self).__init__()
        self.z_size = int(z_size)
        self.net = make_cnn(3*z_size, hidden_sizes, window_size)
        self.data_type = data_type

    def __call__(self, x):
        mapped = self.net(x)

        batch_size = mapped.shape.as_list()[0]
        time_length = mapped.shape.as_list()[1]

        # Obtain mean and precision matrix components
        num_dim = len(mapped.shape.as_list())
        perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        mapped_transposed = tf.transpose(mapped, perm=perm)
        mapped_mean = mapped_transposed[:, :self.z_size]
        mapped_covar = mapped_transposed[:, self.z_size:]

        # tf.nn.sigmoid provides more stable performance on Physionet dataset
        if self.data_type == 'physionet':
            mapped_covar = tf.nn.sigmoid(mapped_covar)
        else:
            mapped_covar = tf.nn.softplus(mapped_covar)

        mapped_reshaped = tf.reshape(mapped_covar, [batch_size, self.z_size, 2*time_length])

        dense_shape = [batch_size, self.z_size, time_length, time_length]
        idxs_1 = np.repeat(np.arange(batch_size), self.z_size*(2*time_length-1))
        idxs_2 = np.tile(np.repeat(np.arange(self.z_size), (2*time_length-1)), batch_size)
        idxs_3 = np.tile(np.concatenate([np.arange(time_length), np.arange(time_length-1)]), batch_size*self.z_size)
        idxs_4 = np.tile(np.concatenate([np.arange(time_length), np.arange(1,time_length)]), batch_size*self.z_size)
        idxs_all = np.stack([idxs_1, idxs_2, idxs_3, idxs_4], axis=1)

        # ~10x times faster on CPU then on GPU
        with tf.device('/cpu:0'):
            # Obtain covariance matrix from precision one
            mapped_values = tf.reshape(mapped_reshaped[:, :, :-1], [-1])
            prec_sparse = tf.sparse.SparseTensor(indices=idxs_all, values=mapped_values, dense_shape=dense_shape)
            prec_sparse = tf.sparse.reorder(prec_sparse)
            prec_tril = tf.sparse_add(tf.zeros(prec_sparse.dense_shape, dtype=tf.float32), prec_sparse)
            eye = tf.eye(num_rows=prec_tril.shape.as_list()[-1], batch_shape=prec_tril.shape.as_list()[:-2])
            prec_tril = prec_tril + eye
            cov_tril = tf.linalg.triangular_solve(matrix=prec_tril, rhs=eye, lower=False)
            cov_tril = tf.where(tf.math.is_finite(cov_tril), cov_tril, tf.zeros_like(cov_tril))

        num_dim = len(cov_tril.shape)
        perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        cov_tril_lower = tf.transpose(cov_tril, perm=perm)
        z_dist = tfd.MultivariateNormalTriL(loc=mapped_mean, scale_tril=cov_tril_lower)
        return z_dist
    
#%% Load, split dataset
""" Load, split dataset """
# --- Path to dataset file ---
path = 'C:\\Users\\Kieron\\Documents\\GitHub\\GP-VAE\\GP-VAE NSG Data\\Data\\original_data.csv'
 
temp_col_names = [
    "ZN01_TOP_Temp", "ZN01_BOTTOM_TEMP", "ZN02_TOP_Temp", "ZN02_BOTTOM_TEMP",
    "ZN03_TOP_Temp", "ZN03_BOTTOM_TEMP", "ZN04_TOP_Temp", "ZN04_BOTTOM_TEMP",
    "ZN05_TOP_Temp", "ZN05_BOTTOM_TEMP", "ZN06_TOP_Temp", "ZN06_BOTTOM_TEMP",
    "ZN07_TOP_Temp", "ZN07_BOTTOM_TEMP", "ZN08_TOP_Temp", "ZN08_BOTTOM_TEMP",
    "ZN09_E01_TEMP", "ZN09_E02_TEMP", "ZN09_E03_TEMP", "ZN09_E04_TEMP",
    "ZN09_E05_TEMP", "ZN09_E06_TEMP", "ZN09_E07_TEMP", "ZN09_E08_TEMP",
    "ZN09_E09_TEMP", "ZN09_E10_TEMP", "ZN09_E11_TEMP", "ZN09_E12_TEMP",
    "ZN09_BOTTOM_TEMP", "ZN10_E01_TEMP", "ZN10_E02_TEMP", "ZN10_E03_TEMP",
    "ZN10_E04_TEMP", "ZN10_E05_TEMP", "ZN10_E06_TEMP", "ZN10_E07_TEMP",
    "ZN10_E08_TEMP", "ZN10_E09_TEMP", "ZN10_E10_TEMP", "ZN10_E11_TEMP",
    "ZN10_E12_TEMP", "ZN10_BOTTOM_TEMP", "ZN11_E01_TEMP", "ZN11_E02_TEMP",
    "ZN11_E03_TEMP", "ZN11_E04_TEMP", "ZN11_E05_TEMP", "ZN11_E06_TEMP",
    "ZN11_E07_TEMP", "ZN11_E08_TEMP", "ZN11_E09_TEMP", "ZN11_E10_TEMP",
    "ZN11_E11_TEMP", "ZN11_E12_TEMP", "ZN11_BOTTOM_TEMP"
]

D = pd.read_csv(path, index_col=0)
X = np.asarray(D.loc[:, temp_col_names].iloc[:771, :])
Y = np.asarray(D.loc[:, "Class"].iloc[:771])
 
train_pc_dataset = 0.8  # Percentage of dataset to be used for training set
n_training_samples = int(X.shape[0] * train_pc_dataset)
n_testing_samples = X.shape[0] - n_training_samples
n_features = X.shape[1]
 

scaler = StandardScaler()
X = scaler.fit_transform(X)
x_train = X[:n_training_samples, :]
x_test = X[n_training_samples:, :]
y_train = Y[:n_training_samples]
y_test = Y[n_training_samples:]

#%% GP Kernels
def rbf_kernel(T, length_scale):
    xs = tf.range(T, dtype=tf.float32)
    xs_in = tf.expand_dims(xs, 0)
    xs_out = tf.expand_dims(xs, 1)
    distance_matrix = tf.math.squared_difference(xs_in, xs_out)
    distance_matrix_scaled = distance_matrix / length_scale ** 2
    kernel_matrix = tf.math.exp(-distance_matrix_scaled)
    return kernel_matrix


def diffusion_kernel(T, length_scale):
    assert length_scale < 0.5, "length_scale has to be smaller than 0.5 for the "\
                               "kernel matrix to be diagonally dominant"
    sigmas = tf.ones(shape=[T, T]) * length_scale
    sigmas_tridiag = tf.linalg.band_part(sigmas, 1, 1)
    kernel_matrix = sigmas_tridiag + tf.eye(T)*(1. - length_scale)
    return kernel_matrix


def matern_kernel(T, length_scale):
    xs = tf.range(T, dtype=tf.float32)
    xs_in = tf.expand_dims(xs, 0)
    xs_out = tf.expand_dims(xs, 1)
    distance_matrix = tf.math.abs(xs_in - xs_out)
    distance_matrix_scaled = distance_matrix / tf.cast(tf.math.sqrt(length_scale), dtype=tf.float32)
    kernel_matrix = tf.math.exp(-distance_matrix_scaled)
    return kernel_matrix


def cauchy_kernel(T, sigma, length_scale):
    xs = tf.range(T, dtype=tf.float32)
    xs_in = tf.expand_dims(xs, 0)
    xs_out = tf.expand_dims(xs, 1)
    distance_matrix = tf.math.squared_difference(xs_in, xs_out)
    distance_matrix_scaled = distance_matrix / length_scale ** 2
    kernel_matrix = tf.math.divide(sigma, (distance_matrix_scaled + 1.))

    alpha = 0.001
    eye = tf.eye(num_rows=kernel_matrix.shape.as_list()[-1])
    return kernel_matrix + alpha * eye

#%% Create, fit encoders
z_size = 32 # Latent space dimensionality - for HMNIST, 256 (down from original 768) - NSG data has ~ 55

diag_encoder = DiagonalEncoder(z_size=z_size)
#joint_encoder = JointEncoder()

diag_encoder.compile()
diag_encoder.fit(x_train, y_train)