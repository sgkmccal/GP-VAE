import tensorflow as tf
import nn_utils
from nn_utils import make_nn, make_cnn, make_2d_cnn
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform

# %% Base Encoders
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



# %% Base Decoders
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
    
class GaussianDecoder(Decoder):
    """ Decoder with Gaussian output distribution (used for SPRITES and Physionet) """
    def __call__(self, x):
        mean = self.net(x)
        var = tf.ones(tf.shape(mean), dtype=tf.float32)
        return tfd.Normal(loc=mean, scale=var)

z_size = 100
diag_enc = DiagonalEncoder(z_size=z_size)

# joint_enc = JointEncoder(z_size=z_size)
# banded_joint_enc = BandedJointEncoder(z_size=z_size)
# print("works")

digit_img = io.imread('Images/2_digit.jpg')

hmnist = np.load('data\hmnist\hmnist_mnar.npz')
# hmnist_10 = hmnist
# print(hmnist.shape)

plt.imshow(digit_img, cmap='gray')
plt.title("Image, original, unmodified")
plt.axis('off')
plt.show()

# Encode the images
digit_encoded = diag_enc(digit_img)
# print("works after digit_encoded")

plt.imshow(digit_encoded.mean().numpy(), cmap='gray')
plt.title("Encoded image")
plt.axis('off')
plt.show()

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

bernoulli_decoder = BernoulliDecoder(28*28)

# bernoulli_reconstruction = tf.reshape(bernoulli_decoder(jax_scaled_encoded_reshaped).mean(), (180,256))
bernoulli_output = bernoulli_decoder(digit_encoded.mean())
# print("Bernoulli output shape:", bernoulli_output.mean().shape)  # Check the shape here

# print(bernoulli_output.mean())

plt.imshow(tf.reshape(bernoulli_output.mean()[0], (28,28)))
plt.axis('off')
plt.show()


# code breaks here
# bernoulli_output_reshaped = tf.reshape(bernoulli_output.mean(), (28,28))

# gaussian_output = tf.reshape(gaussian_decoder(jax_scaled_encoded_reshaped).mean(),(180,256))

# plt.imshow(bernoulli_output_reshaped.mean().numpy())
# plt.title("Decoder output, reshaped to size of original input image")
# plt.axis('off')
# plt.show()

# plt.imshow(gaussian_output.mean().numpy())
# plt.axis('off')
# plt.show()