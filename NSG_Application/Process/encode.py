import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic, Kernel
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import pandas as pd
from sklearn.preprocessing import StandardScaler

# NN Utils
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

# Encoders
# class BaseEncoder(tf.keras.Model):
#     # Basic Dense FF Encoder
#     def __init__(self, latent_dim):
#         super(BaseEncoder, self).__init__()
#         self.encoder = tf.keras.Sequential([
#             tf.keras.layers.InputLayer(input_shape=(inputs.shape[1],)),  # Input: x_train features
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(latent_dim, activation='relu')  # Encoded representation
#         ])
#         self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')  # binary output - y_train value

#     def call(self, inputs):
#         encoded = self.encoder(inputs)
#         return self.output_layer(encoded)

class BaseEncoder(tf.keras.Model):
    # Basic Dense (Feed-Forward) Encoder
    def __init__(self, latent_dim=32, layers=(64,64), batch_size=32):
        super(BaseEncoder, self).__init__()

        self.batch_size = batch_size

        # Validate that `layers` is a tuple or list of integers
        if not isinstance(layers, (list, tuple)) or not all(isinstance(l, int) for l in layers):
            raise ValueError(f"`layers` must be a list or tuple of integers, got {layers}.")
        
        # Create a Sequential model for the encoder
        self.encoder = tf.keras.Sequential()

        # Add each layer with the specified number of neurons
        for num_neurons in layers:
            self.encoder.add(tf.keras.layers.Dense(num_neurons, activation='relu'))

        self.encoder.add(tf.keras.layers.Dense(latent_dim, activation='relu'))

    def __call__(self, inputs):
        inputs = tf.expand_dims(inputs, axis=0)
        inputs_batched = inputs.batch(self.batch_size)
        return self.encoder(inputs)

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

    def __call__(self, x, training=False):
        # mapped = self.net(x, training=training)
        # return tfd.MultivariateNormalDiag(
        #   loc=mapped[..., :self.z_size],
        #   scale_diag=tf.nn.softplus(mapped[..., self.z_size:]))

        # changing return (output) to be mean
        mapped = self.net(x, training=training)
        loc = mapped[..., :self.z_size]  # Mean
        scale_diag = tf.nn.softplus(mapped[..., self.z_size:])  # Variance
        return loc 

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

    def __call__(self, x, training=False):
        # requires 3D input (batch_size, time_steps=1, features) so adding sequencelength = 1, since using single feature vector per sample
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)

        mapped = self.net(x, training=training)
        if self.transpose:
            num_dim = len(x.shape.as_list())
            perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
            mapped = tf.transpose(mapped, perm=perm)
            return tfd.MultivariateNormalDiag(
                    loc=mapped[..., :self.z_size, :],
                    scale_diag=tf.nn.softplus(mapped[..., self.z_size:, :]))
        # return tfd.MultivariateNormalDiag(
        #             loc=mapped[..., :self.z_size],
        #             scale_diag=tf.nn.softplus(mapped[..., self.z_size:]))

        loc = mapped[..., :self.z_size]  # Extract the mean
        scale_diag = tf.nn.softplus(mapped[..., self.z_size:])  # Extract the variance
        return loc  # Output the mean tensor for training


x = np.linspace(0,10,100)
be = BaseEncoder(8, (64, 64, 32, 32, 16, 16))
x_be = be(x)
print(x, "\n", x_be)