import sys
import os
import time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from absl import app
# from absl import flags
sys.path.append("..")
from lib.models import *
# import debug
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from lib import nn_utils
import tensorflow as tf


# ImagePreprocessor
class ImagePreprocessor(tf.keras.Model):
    def __init__(self, image_shape, hidden_sizes=(256, ), kernel_size=3.):
        """ Decoder parent class without specified output distribution
            :param image_shape: input image size
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
            :param kernel_size: kernel/filter width and height
        """
        super(ImagePreprocessor, self).__init__()
        self.image_shape = image_shape
        self.net = make_2d_cnn(image_shape[-1], hidden_sizes, kernel_size)

    def __call__(self, x):
        return self.net(x)
    
# Cauchy, RBF, Diffusion, Matern kernels
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

# make_nn, make_cnn, make 2d_cnn
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

# Base Decoder, BernoulliDecoder
class Decoder(tf.keras.Model):
    def __init__(self, output_size, hidden_sizes=(64, 64)):
        """ Decoder parent class with no specified output distribution
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
        """
        super(Decoder, self).__init__()
        self.net = nn_utils.make_nn(output_size, hidden_sizes)

    def __call__(self, x):
        pass

class BernoulliDecoder(Decoder):
    """ Decoder with Bernoulli output distribution (used for HMNIST) """
    def __call__(self, x):
        mapped = self.net(x)
        return tfd.Bernoulli(logits=mapped)
    
# Diagonal Encoder, JointEncoder
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

# VAE, HI-VAE, GP-VAE model classes
class VAE(tf.keras.Model):
    def __init__(self, latent_dim, data_dim, time_length,
                 encoder_sizes=(64, 64), encoder=DiagonalEncoder,
                 decoder_sizes=(64, 64), decoder=BernoulliDecoder,
                 image_preprocessor=None, beta=1.0, M=1, K=1, **kwargs):
        """ Basic Variational Autoencoder with Standard Normal prior
            :param latent_dim: latent space dimensionality
            :param data_dim: original data dimensionality
            :param time_length: time series duration
            
            :param encoder_sizes: layer sizes for the encoder network
            :param encoder: encoder model class {Diagonal, Joint, BandedJoint}Encoder
            :param decoder_sizes: layer sizes for the decoder network
            :param decoder: decoder model class {Bernoulli, Gaussian}Decoder
            
            :param image_preprocessor: 2d-convolutional network used for image data preprocessing
            :param beta: tradeoff coefficient between reconstruction and KL terms in ELBO
            :param M: number of Monte Carlo samples for ELBO estimation
            :param K: number of importance weights for IWAE model (see: https://arxiv.org/abs/1509.00519)
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.time_length = time_length

        self.encoder = encoder(latent_dim, encoder_sizes, **kwargs)
        self.decoder = decoder(data_dim, decoder_sizes)
        self.preprocessor = image_preprocessor

        self.beta = beta
        self.K = K
        self.M = M

    def encode(self, x):
        x = tf.identity(x)  # in case x is not a Tensor already...
        if self.preprocessor is not None:
            x_shape = x.shape.as_list()
            new_shape = [x_shape[0] * x_shape[1]] + list(self.preprocessor.image_shape)
            x_reshaped = tf.reshape(x, new_shape)
            x_preprocessed = self.preprocessor(x_reshaped)
            x = tf.reshape(x_preprocessed, x_shape)
        return self.encoder(x)

    def decode(self, z):
        z = tf.identity(z)  # in case z is not a Tensor already...
        return self.decoder(z)

    def __call__(self, inputs):
        return self.decode(self.encode(inputs).sample()).sample()

    def generate(self, noise=None, num_samples=1):
        if noise is None:
            noise = tf.random_normal(shape=(num_samples, self.latent_dim))
        return self.decode(noise)
    
    def _get_prior(self):
        if self.prior is None:
            self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim, dtype=tf.float32),
                                                    scale_diag=tf.ones(self.latent_dim, dtype=tf.float32))
        return self.prior

    def compute_nll(self, x, y=None, m_mask=None):
        # Used only for evaluation
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        if y is None: y = x

        z_sample = self.encode(x).sample()
        x_hat_dist = self.decode(z_sample)
        nll = -x_hat_dist.log_prob(y)  # shape=(BS, TL, D)
        nll = tf.where(tf.math.is_finite(nll), nll, tf.zeros_like(nll))
        if m_mask is not None:
            m_mask = tf.cast(m_mask, tf.bool)
            nll = tf.where(m_mask, nll, tf.zeros_like(nll))  # !!! inverse mask, set zeros for observed
        return tf.reduce_sum(nll)

    def compute_mse(self, x, y=None, m_mask=None, binary=False):
        # Used only for evaluation
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        if y is None: y = x

        z_mean = self.encode(x).mean()
        x_hat_mean = self.decode(z_mean).mean()  # shape=(BS, TL, D)
        if binary:
            x_hat_mean = tf.round(x_hat_mean)
        mse = tf.math.squared_difference(x_hat_mean, y)
        if m_mask is not None:
            m_mask = tf.cast(m_mask, tf.bool)
            mse = tf.where(m_mask, mse, tf.zeros_like(mse))  # !!! inverse mask, set zeros for observed
        return tf.reduce_sum(mse)

    def _compute_loss(self, x, m_mask=None, return_parts=False):
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        x = tf.tile(x, [self.M * self.K, 1, 1])  # shape=(M*K*BS, TL, D)

        if m_mask is not None:
            m_mask = tf.identity(m_mask)  # in case m_mask is not a Tensor already...
            m_mask = tf.tile(m_mask, [self.M * self.K, 1, 1])  # shape=(M*K*BS, TL, D)
            m_mask = tf.cast(m_mask, tf.bool)

        pz = self._get_prior()
        qz_x = self.encode(x)
        z = qz_x.sample()
        px_z = self.decode(z)

        nll = -px_z.log_prob(x)  # shape=(M*K*BS, TL, D)
        nll = tf.where(tf.math.is_finite(nll), nll, tf.zeros_like(nll))
        if m_mask is not None:
            nll = tf.where(m_mask, tf.zeros_like(nll), nll)  # if not HI-VAE, m_mask is always zeros
        nll = tf.reduce_sum(nll, [1, 2])  # shape=(M*K*BS)

        if self.K > 1:
            kl = qz_x.log_prob(z) - pz.log_prob(z)  # shape=(M*K*BS, TL or d)
            kl = tf.where(tf.is_finite(kl), kl, tf.zeros_like(kl))
            kl = tf.reduce_sum(kl, 1)  # shape=(M*K*BS)

            weights = -nll - kl  # shape=(M*K*BS)
            weights = tf.reshape(weights, [self.M, self.K, -1])  # shape=(M, K, BS)

            elbo = reduce_logmeanexp(weights, axis=1)  # shape=(M, 1, BS)
            elbo = tf.reduce_mean(elbo)  # scalar
        else:
            # if K==1, compute KL analytically
            kl = self.kl_divergence(qz_x, pz)  # shape=(M*K*BS, TL or d)
            kl = tf.where(tf.math.is_finite(kl), kl, tf.zeros_like(kl))
            kl = tf.reduce_sum(kl, 1)  # shape=(M*K*BS)

            elbo = -nll - self.beta * kl  # shape=(M*K*BS) K=1
            elbo = tf.reduce_mean(elbo)  # scalar

        if return_parts:
            nll = tf.reduce_mean(nll)  # scalar
            kl = tf.reduce_mean(kl)  # scalar
            return -elbo, nll, kl
        else:
            return -elbo

    def compute_loss(self, x, m_mask=None, return_parts=False):
        del m_mask
        return self._compute_loss(x, return_parts=return_parts)

    def kl_divergence(self, a, b):
        return tfd.kl_divergence(a, b)

    def get_trainable_vars(self):
        self.compute_loss(tf.random.normal(shape=(1, self.time_length, self.data_dim), dtype=tf.float32),
                          tf.zeros(shape=(1, self.time_length, self.data_dim), dtype=tf.float32))
        return self.trainable_variables
      
class HI_VAE(VAE):
    """ HI-VAE model, where the reconstruction term in ELBO is summed only over observed components """
    def compute_loss(self, x, m_mask=None, return_parts=False):
        return self._compute_loss(x, m_mask=m_mask, return_parts=return_parts)

class GP_VAE(HI_VAE):
    def __init__(self, *args, kernel="cauchy", sigma=1., length_scale=1.0, kernel_scales=1, **kwargs):
        """ Proposed GP-VAE model with Gaussian Process prior
            :param kernel: Gaussial Process kernel ["cauchy", "diffusion", "rbf", "matern"]
            :param sigma: scale parameter for a kernel function
            :param length_scale: length scale parameter for a kernel function
            :param kernel_scales: number of different length scales over latent space dimensions
        """
        super(GP_VAE, self).__init__(*args, **kwargs)
        self.kernel = kernel
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales

        if isinstance(self.encoder, JointEncoder):
            self.encoder.transpose = True

        # Precomputed KL components for efficiency
        self.pz_scale_inv = None
        self.pz_scale_log_abs_determinant = None
        self.prior = None

    def decode(self, z):
        num_dim = len(z.shape)
        assert num_dim > 2
        perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        return self.decoder(tf.transpose(z, perm=perm))

    def _get_prior(self):
        if self.prior is None:
            # Compute kernel matrices for each latent dimension
            kernel_matrices = []
            for i in range(self.kernel_scales):
                if self.kernel == "rbf":
                    kernel_matrices.append(rbf_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "diffusion":
                    kernel_matrices.append(diffusion_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "matern":
                    kernel_matrices.append(matern_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "cauchy":
                    kernel_matrices.append(cauchy_kernel(self.time_length, self.sigma, self.length_scale / 2**i))

            # Combine kernel matrices for each latent dimension
            tiled_matrices = []
            total = 0
            for i in range(self.kernel_scales):
                if i == self.kernel_scales-1:
                    multiplier = self.latent_dim - total
                else:
                    multiplier = int(np.ceil(self.latent_dim / self.kernel_scales))
                    total += multiplier
                tiled_matrices.append(tf.tile(tf.expand_dims(kernel_matrices[i], 0), [multiplier, 1, 1]))
            kernel_matrix_tiled = np.concatenate(tiled_matrices)
            assert len(kernel_matrix_tiled) == self.latent_dim

            self.prior = tfd.MultivariateNormalFullCovariance(
                loc=tf.zeros([self.latent_dim, self.time_length], dtype=tf.float32),
                covariance_matrix=kernel_matrix_tiled)
        return self.prior

    def kl_divergence(self, a, b):
        """ Batched KL divergence `KL(a || b)` for multivariate Normals.
            See https://github.com/tensorflow/probability/blob/master/tensorflow_probability
                       /python/distributions/mvn_linear_operator.py
            It's used instead of default KL class in order to exploit precomputed components for efficiency
        """

        def squared_frobenius_norm(x):
            """Helper to make KL calculation slightly more readable."""
            return tf.reduce_sum(tf.square(x), axis=[-2, -1])

        def is_diagonal(x):
            """Helper to identify if `LinearOperator` has only a diagonal component."""
            return (isinstance(x, tf.linalg.LinearOperatorIdentity) or
                    isinstance(x, tf.linalg.LinearOperatorScaledIdentity) or
                    isinstance(x, tf.linalg.LinearOperatorDiag))

        if is_diagonal(a.scale) and is_diagonal(b.scale):
            # Using `stddev` because it handles expansion of Identity cases.
            b_inv_a = (a.stddev() / b.stddev())[..., tf.newaxis]
        else:
            if self.pz_scale_inv is None:
                self.pz_scale_inv = tf.linalg.inv(b.scale.to_dense())
                self.pz_scale_inv = tf.where(tf.math.is_finite(self.pz_scale_inv),
                                             self.pz_scale_inv, tf.zeros_like(self.pz_scale_inv))

            if self.pz_scale_log_abs_determinant is None:
                self.pz_scale_log_abs_determinant = b.scale.log_abs_determinant()

            a_shape = a.scale.shape
            if len(b.scale.shape) == 3:
                _b_scale_inv = tf.tile(self.pz_scale_inv[tf.newaxis], [a_shape[0]] + [1] * (len(a_shape) - 1))
            else:
                _b_scale_inv = tf.tile(self.pz_scale_inv, [a_shape[0]] + [1] * (len(a_shape) - 1))

            b_inv_a = _b_scale_inv @ a.scale.to_dense()

        # ~10x times faster on CPU then on GPU
        with tf.device('/cpu:0'):
            kl_div = (self.pz_scale_log_abs_determinant - a.scale.log_abs_determinant() +
                      0.5 * (-tf.cast(a.scale.domain_dimension_tensor(), a.dtype) +
                      squared_frobenius_norm(b_inv_a) + squared_frobenius_norm(
                      b.scale.solve((b.mean() - a.mean())[..., tf.newaxis]))))
        return kl_div

""" PARAMETERS """
seed = 451 # RNG SEED

# Set experiment name, save location, etc.
basedir = "models"
model_type = "hmnist"
exp_name = "debug"
timestamp = datetime.now().strftime("%y%m%d")
full_exp_name = "{}_{}".format(timestamp, exp_name)
outdir = os.path.join(basedir, full_exp_name)
if not os.path.exists(outdir): os.mkdir(outdir)
checkpoint_prefix = os.path.join(outdir, "ckpt")

def main(argv):
    del argv
    
    
    """Set rng params"""
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    print("Full exp name: ", full_exp_name)
    
    """ generic parameters """
    learning_rate = 1e-3  # training lr
    gradient_clip = 1e4  # max value for grads during backprop
    num_steps = 0 # n training steps
    cnn_kernel_size = 3 # size of filter sliding over image for cnn preprocessor
    cnn_sizes = [256]  # num. filters for layers of cnn preprocessor
    banded_covar = False # whether or not to use banded covariance matrix for 
                         # output of inference network
    testing = False
    print_interval = 0 # interval for printing loss and saving model during training
    M=1 # n samples for ELBO estimation
    K=1 # n importance sampling weights

    """ HMNIST parameters """
    latent_dim = 256
    encoder_sizes = [256,256]
    decoder_sizes = [256,256,256]
    num_epochs = 5
    sigma = 1.0
    beta = 0.8
    length_scale = 1.0
    window_size = 3             
    batch_size = 64
    
    """ Variables only used in GP-VAE model type """
    kernel = "cauchy" # cauchy (default), diffusion, matern, rbf
    kernel_scales = 1 # num different lengthscale sigmas 
    
    """ define autoencoder layer sizes"""
    encoder_sizes = [256,256] # 2 layers of 256 neurons
    decoder_sizes = [256,256,256]
    
    """ load dataset """
    data_type = "hmnist"
    data_dir = "data\hmnist\hmnist_mnar.npz"   # path to (H)MNIST data
    data = np.load(data_dir)
    data_dim = 784 # num features in a sample, 28x28 px = 784
    time_length = 10
    num_classes = 10
    
    decoder = BernoulliDecoder
    img_shape = (28,28,1) # shape of image when reconstructed === input image shape
    

    """ partition dataset """
    use_full_dataset = True
    val_split = 50000 # num samples from df to use as training samples
    subset_n_train = 500
    subset_n_val = 100
    
    if use_full_dataset == True:    
        x_train_full = data['x_train_full']
        x_train_miss = data['x_train_miss']
        m_train_miss = data['m_train_miss']
        y_train = data['y_train']
        
        # paper uses "validation" in place of regular "test" set
        x_val_full = x_train_full[val_split:]   # validation set of x
        x_val_miss = x_train_miss[val_split:]
        m_val_miss = m_train_miss[val_split:]
        y_val = y_train[val_split:]
        
        x_train_full = x_train_full[:val_split]
        x_train_miss = x_train_miss[:val_split]
        m_train_miss = m_train_miss[:val_split]
        y_train = y_train[:val_split]

    else:
        x_train_full = data['x_train_full']
        x_train_miss = data['x_train_miss']
        m_train_miss = data['m_train_miss']
        y_train = data['y_train']
        
        # paper uses "validation" in place of regular "test" set
        x_val_full = x_train_full[subset_n_train:subset_n_val]   # validation set of x
        x_val_miss = x_train_miss[subset_n_train:subset_n_val]
        m_val_miss = m_train_miss[subset_n_train:subset_n_val]
        y_val = y_train[subset_n_train:subset_n_val]
        
        x_train_full = x_train_full[:subset_n_train]
        x_train_miss = x_train_miss[:subset_n_train]
        m_train_miss = m_train_miss[:subset_n_train]
        y_train = y_train[:subset_n_train]
    
    # convert to tf tensors
    tf_x_train_miss = tf.data.Dataset.from_tensor_slices((x_train_miss, m_train_miss))\
                     .shuffle(len(x_train_miss)).batch(batch_size=batch_size).repeat()
    tf_x_val_miss = tf.data.Dataset.from_tensor_slices((x_val_miss, m_val_miss)).batch(batch_size).repeat()
    tf_x_val_miss = tf.compat.v1.data.make_one_shot_iterator(tf_x_val_miss)
    
    
    """ construct Conv2D preprocessor for image data """
    print("Using CNN preprocessor")
    image_preprocessor = ImagePreprocessor(img_shape, 
                                           cnn_sizes, 
                                           cnn_kernel_size)
    
    
    """ Construct model """
    """ VAE """
    
    """ GP-VAE """
    # using jointencoder instead of bandedjointencoder, since banded_covar = false
    encoder = JointEncoder
    model = GP_VAE(latent_dim=latent_dim,
                   data_dim=data_dim,
                   time_length=time_length,
                   encoder_sizes=encoder_sizes, encoder=encoder,
                   decoder_sizes=decoder_sizes, decoder=decoder,
                   kernel=kernel, sigma=sigma,
                   length_scale=length_scale, kernel_scales=kernel_scales,
                   image_preprocessor=image_preprocessor, window_size=window_size,
                   beta=beta, M=M, K=K, data_type=data_type
                   )
    
    
    """ Training preparation """
    print("Training...")
    _ = tf.compat.v1.train.get_or_create_global_step()
    trainable_vars = model.get_trainable_vars()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    
    print("Encoder: ", model.encoder.net.summary())
    print("Decoder: ", model.decoder.net.summary())
    
    if model.preprocessor is not None:
        print("Preprocessor: ", model.preprocessor.net.summary())
        saver = tf.compat.v1.train.Checkpoint(optimizer=optimizer, encoder=model.encoder.net, 
                                              decoder=model.decoder.net, preprocessor=model.preprocessor.net, 
                                              optimizer_step=tf.compat.v1.train.get_or_create_global_step())
    else:
        saver = tf.compat.v1.train.Checkpoint(optimizer=optimizer, 
                                              encoder=model.encoder.net, decoder=model.decoder.net,
                                              optimizer_step=tf.compat.v1.train.get_or_create_global_step())
    summary_writer = tf.contrib.summary.create_file_writer(outdir, flush_millis=10000)
    
    if num_steps == 0:
        num_steps = num_epochs * len(x_train_miss) // batch_size
    else:
        num_steps = num_steps
    
    if print_interval == 0:
        print_interval = num_steps // num_epochs
    
    
    """ Training stage """
    losses_train = []
    losses_val = []
    
    t0 = time.time()
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        for i, (x_seq, m_seq) in enumerate(tf_x_train_miss.take(num_steps)):
            try:
                with tf.GradientTape() as tape:
                    tape.watch(trainable_vars)
                    loss = model.compute_loss(x_seq, m_mask=m_seq)
                    losses_train.append(loss.numpy())
                grads = tape.gradient(loss, trainable_vars)
                grads = [np.nan_to_num(grad) for grad in grads]
                grads, global_norm = tf.clip_by_global_norm(grads, gradient_clip)
                optimizer.apply_gradients(zip(grads, trainable_vars),
                                          global_step=tf.compat.v1.train.get_or_create_global_step())
    
                # Print intermediate results
                if i % print_interval == 0:
                    print("================================================")
                    print("Learning rate: {} | Global gradient norm: {:.2f}".format(optimizer._lr, global_norm))
                    print("Step {}) Time = {:2f}".format(i, time.time() - t0))
                    loss, nll, kl = model.compute_loss(x_seq, m_mask=m_seq, return_parts=True)
                    print("Train loss = {:.3f} | NLL = {:.3f} | KL = {:.3f}".format(loss, nll, kl))

                    saver.save(checkpoint_prefix)
                    tf.contrib.summary.scalar("loss_train", loss)
                    tf.contrib.summary.scalar("kl_train", kl)
                    tf.contrib.summary.scalar("nll_train", nll)

                    # Validation loss
                    x_val_batch, m_val_batch = tf_x_val_miss.get_next()
                    val_loss, val_nll, val_kl = model.compute_loss(x_val_batch, m_mask=m_val_batch, return_parts=True)
                    losses_val.append(val_loss.numpy())
                    print("Validation loss = {:.3f} | NLL = {:.3f} | KL = {:.3f}".format(val_loss, val_nll, val_kl))

                    tf.contrib.summary.scalar("loss_val", val_loss)
                    tf.contrib.summary.scalar("kl_val", val_kl)
                    tf.contrib.summary.scalar("nll_val", val_nll)
                    
                    # Datatype = hmnist
                    # Draw reconstructed images
                    x_hat = model.decode(model.encode(x_seq).sample()).mean()
                    tf.contrib.summary.image("input_train", tf.reshape(x_seq, [-1]+list(img_shape)))
                    tf.contrib.summary.image("reconstruction_train", tf.reshape(x_hat, [-1]+list(img_shape)))
    
                    t0 = time.time()
                    
            except KeyboardInterrupt:
                saver.save(checkpoint_prefix)
                if debug:
                    import ipdb
                    ipdb.set_trace()
                break
            
    """ Evaluation stage """
    print("Evaluation...")
    
    # splitting data on batches
    x_val_miss_batches = np.array_split(x_val_miss, batch_size, axis=0)
    x_val_full_batches = np.array_split(x_val_full, batch_size, axis=0)
    m_val_batches = np.array_split(m_val_miss, batch_size, axis=0)
    get_val_batches = lambda: zip(x_val_miss_batches, x_val_full_batches, m_val_batches)
    
    # compute NLL and MSE on missing values
    # n_missings = m_val_artificial.sum() if data_type == 'physionet' else m_val_miss.sum()
    n_missings = m_val_miss.sum()
    nll_miss = np.sum([model.compute_nll(x, y=y, m_mask=m).numpy()
                      for x, y, m in get_val_batches()]) / n_missings
    mse_miss = np.sum([model.compute_mse(x, y=y, m_mask=m, binary=data_type=="hmnist").numpy()
                      for x, y, m in get_val_batches()]) / n_missings
    print("NLL miss: {:.4f}".format(nll_miss))
    print("MSE miss: {:.4f}".format(mse_miss))
    
    # saving imputed values
    z_mean = [model.encode(x_batch).mean().numpy() for x_batch in x_val_miss_batches]
    np.save(os.path.join(outdir, "z_mean"), np.vstack(z_mean))
    x_val_imputed = np.vstack([model.decode(z_batch).mean().numpy() for z_batch in z_mean])
    np.save(os.path.join(outdir, "imputed_no_gt"), x_val_imputed)
    
    # impute gt observed values
    x_val_imputed[m_val_miss == 0] = x_val_miss[m_val_miss == 0]
    np.save(os.path.join(outdir, "imputed"), x_val_imputed)
    
    # datatype = hmnist, ...:
    # AUROC evaluation using Logistic Regression
    # Area Under the Received Operating Characteristic
    x_val_imputed = np.round(x_val_imputed)
    x_val_imputed = x_val_imputed.reshape([-1, time_length * data_dim])

    cls_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-10, max_iter=10000)
    val_split = len(x_val_imputed) // 2

    cls_model.fit(x_val_imputed[:val_split], y_val[:val_split])
    probs = cls_model.predict_proba(x_val_imputed[val_split:])

    auprc = average_precision_score(np.eye(num_classes)[y_val[val_split:]], probs)
    auroc = roc_auc_score(np.eye(num_classes)[y_val[val_split:]], probs)
    print("AUROC: {:.4f}".format(auroc))
    print("AUPRC: {:.4f}".format(auprc))
    
    
    """ Visualise reconstructions """
    img_index=0 
    img_shape = (28,28)
    cmap='gray'
    
    fig, axes = plt.subplots(nrows=3, ncols=x_val_miss.shape[1], 
                             figsize=(2*x_val_miss.shape[1], 6))

    x_hat = model.decode(model.encode(x_val_miss[img_index: img_index+1]).mean()).mean().numpy()
    seqs = [x_val_miss[img_index:img_index+1], x_hat, x_val_full[img_index:img_index+1]]

    for axs, seq in zip(axes, seqs):
        for ax, img in zip(axs, seq[0]):
            ax.imshow(img.reshape(img_shape), cmap=cmap)
            ax.axis('off')

    suptitle = model_type + f" reconstruction, NLL missing = {mse_miss}"
    fig.suptitle(suptitle, size=18)
    fig.savefig(os.path.join(outdir,  data_type + "_reconstruction.pdf"))
    
    results_all = [seed, model_type, data_type, kernel, beta, latent_dim,
                   num_epochs, batch_size, learning_rate, window_size,
                   kernel_scales, sigma, length_scale,
                   len(encoder_sizes), encoder_sizes[0] if len(encoder_sizes) > 0 else 0,
                   len(decoder_sizes), decoder_sizes[0] if len(decoder_sizes) > 0 else 0,
                   cnn_kernel_size, cnn_sizes,
                   nll_miss, mse_miss, losses_train[-1], losses_val[-1], auprc, auroc, testing, data_dir]
    
    with open(os.path.join(outdir, "results.tsv"), "w") as outfile:
        outfile.write("seed\tmodel\tdata\tkernel\tbeta\tz_size\tnum_epochs"
                      "\tbatch_size\tlearning_rate\twindow_size\tkernel_scales\t"
                      "sigma\tlength_scale\tencoder_depth\tencoder_width\t"
                      "decoder_depth\tdecoder_width\tcnn_kernel_size\t"
                      "cnn_sizes\tNLL\tMSE\tlast_train_loss\tlast_val_loss\tAUPRC\tAUROC\ttesting\tdata_dir\n")
        outfile.write("\t".join(map(str, results_all)))

    with open(os.path.join(outdir, "training_curve.tsv"), "w") as outfile:
        outfile.write("\t".join(map(str, losses_train)))
        outfile.write("\n")
        outfile.write("\t".join(map(str, losses_val)))

    print("Training finished.")


if __name__ == '__main__':
    app.run(main)









