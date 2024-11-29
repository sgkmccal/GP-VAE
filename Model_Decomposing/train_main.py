import tensorflow as tf
import nn_utils
from nn_utils import make_nn, make_cnn, make_2d_cnn
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
from keras.datasets import mnist

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

class Decoder(tf.keras.Model):
    def __init__(self, output_size, hidden_sizes=(64, 64)):
        """ Decoder parent class with no specified output distribution
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
        """
        super(Decoder, self).__init__()
        self.net = make_nn(output_size, hidden_sizes)

    def __call__(self, x):
        pass

class BernoulliDecoder(Decoder):
    """ Decoder with Bernoulli output distribution (used for HMNIST) """
    def __call__(self, x):
        mapped = self.net(x)
        return tfd.Bernoulli(logits=mapped)
    
class GaussianDecoder(Decoder):
    """ Decoder with Gaussian output distribution (used for SPRITES and Physionet) """
    def __call__(self, x):
        mean = self.net(x)
        var = tf.ones(tf.shape(mean), dtype=tf.float32)
        return tfd.Normal(loc=mean, scale=var)

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

class ImagePreprocessor(tf.keras.Model):
    def __init__(self, image_shape, hidden_sizes=256, kernel_size=3):
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

""" ---- Load Data ---- """
(x_train, y_train), (x_test, y_test) = mnist.load_data()


"""  ---- Data(-specific) properties ---- """
data_dim = 784 # dimensionality of the input data, 28x28 = 784 for mnist
time_length = 10 # Unknown meaning , 10 for HMNIST so using same for MNIST
encoderClass = DiagonalEncoder # ref. pointing to Encoder class
decoderClass = BernoulliDecoder # ref. pointing to Decoder class
img_shape = (28,28,1) # shape of image 28px by 28px
val_split = 50000 # (assumed) num samples to train with
latent_representation_size = 256 # z_size
decoder_output_size = 28*28 # size decoded image should be, 784 = original image's size = 28x28 px
# encoder_layer_sizes = 
# decoder_hidden_layer_sizes =  # tuple (form is required) of hidden layer sizes eg (size1, size2, size3, ...)

""" ---- GP-VAE-specific properties ---- """
window_size = 3 # window size for inference CNN
sigma = 1.0 # sigma value for GP prior
lengthscale = 2.0 # lengthscale value for GP prior
beta = 0.8 # factor to weigh KL term (similar to beta-VAE (?))

num_epochs = 20 # number of training epochs

kernel = 'cauchy' # kernel used for GP prior
kernel_scales = 1 # num of different length scales sigma for GP prior

""" ---- Misc properties ---- """
learning_rate = 1e-3 #  lr. for training
gradient_clip = 1e4 # max. global gradient norm for gradient clipping during training
num_steps = 0 # num. training steps, if > 0, will overwrite num_epochs
print_interval = 0 # interval for printing loss and saving model while training
exp_name = "debug"
# basedir="" # base directory where models are stored
# data_dir = ""
# data_type = None
# seed = 1337 # seed for rng
# model_type = 
cnn_kernel_size = 3 # kernel size for CNN preprocessor
testing = False # use actual test set for testing
banded_covar = False # use banded covariance matrix instead of diagonal one for output of inference network
                     # ignored if model_type != gp-vae
batch_size = 64 # training batch size

num_samples_ELBO_estimation = 1 # number of samples for ELBO estimation
num_importance_sampling_weights = 1 # number of importance sampling weights to use



""" ---- Isolated encoder and decoder ---- """
diag_enc = DiagonalEncoder(z_size=latent_representation_size)
bernoulli_dec = BernoulliDecoder(output_size=decoder_output_size, hidden_sizes=(256,256,256))

# print(x_train.shape)

train_subset = x_train[:5000]
# print("train_subset shape: ", train_subset.shape)

""" ---- Encoded versions of MNIST images ---- """
mnist_encoded = diag_enc(x_train.reshape(60000, 784))


image_preprocessor = ImagePreprocessor(image_shape=(28, 28, 1),
                                       hidden_sizes = [256], # "number of filters for layers of CNN preprocessor", originally set to 256
                                       kernel_size = 3 # "kernel size for CNN preprocessor", originally set to 3
                                       ) 

""" ---- Creating VAE model ---- """
model = VAE(latent_dim=latent_representation_size, # dimensionality of latent space 
            data_dim=decoder_output_size,                     # dims of inputs, 28x28=784
            time_length=10,        # !!!: ???? setting to fixed int for now 
            encoder_sizes=(64,64),   # sizes of encoder layers, i.e. how many neurons in each layer
            # encoder=DiagonalEncoder, 
            encoder = DiagonalEncoder,    # set to class or instance of class? original code uses class itself?
            decoder_sizes=(256,256,256),  # sizes of decoder layers (256,256,256)
            # decoder=decoder,
            decoder = BernoulliDecoder,       
            image_preprocessor=image_preprocessor, 
            # window_size=3, # "window size for inference CNN: ignored if model type != gp-vae", def.=3
            beta=0.8, # factor to weight KL term (similar to beta-VAE) def=0.8
            M=1,  # num samples for ELBO estimation, def=1
            K=1)  # num importance sampling weights, def=1

""" ---- Experiment name ----- """
# import datetime
# timestamp = datetime.now().strftime("%y%m%d")
# full_exp_name = "{}_{}".format(timestamp, exp_name)
# outdir = os.path.join(FLAGS.basedir, full_exp_name)
# if not os.path.exists(outdir): os.mkdir(outdir)
# checkpoint_prefix = os.path.join(outdir, "ckpt")
# print("Full exp name: ", full_exp_name)


""" ---- Training model ---- """
_ = tf.compat.v1.train.get_or_create_global_step() # Returns and create (if necessary) the global step tensor
trainable_vars = model.get_trainable_vars()
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

print("Encoder: ", model.encoder.net.summary())
print("Decoder: ", model.decoder.net.summary())

# this version is for data that uses a preprocessor (hmnist, sprites), there is a different set of code for physionet
print("Preprocessor: ", model.preprocessor.net.summary())
saver = tf.compat.v1.train.Checkpoint(optimizer = optimizer,
                                      encoder=model.encoder.net,
                                      decoder=model.decoder.net,
                                      preprocessor=model.preprocessor.net,
                                      optimizer_step=tf.compat.v1.train.get_or_create_global_step())

