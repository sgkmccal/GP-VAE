import tensorflow as tf
from lib import nn_utils
from lib.nn_utils import make_nn, make_cnn, make_2d_cnn
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform

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
    
z_size = 100
diag_enc = DiagonalEncoder(z_size=z_size)

digit_img = io.imread(".\Images\\2_digit.jpg")

plt.imshow(digit_img, cmap='gray')
plt.title("Image, original, unmodified")
plt.axis('off')
plt.show()

# Encode the images
digit_encoded = diag_enc(digit_img)
print("works after digit_encoded")

plt.imshow(digit_encoded.mean().numpy(), cmap='gray')
plt.title("Encoded image")
plt.axis('off')
plt.show()

class Decoder_2828(tf.keras.Model):
    # Modified base decoder class for MNIST images of size (28,28) instead of HMNIST (64,64)
    def __init__(self, output_size, hidden_sizes=(28,28)):
        """ Decoder parent class with no specified output distribution
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
        """
        super(Decoder_2828, self).__init__()
        self.net = nn_utils.make_nn(output_size, hidden_sizes)

    def __call__(self, x):
        pass

class BernoulliDecoder_2828(Decoder_2828):
    # Modified Bernoulli decoder for (28,28) images
    """ Decoder with Bernoulli output distribution (used for HMNIST) """
    def __call__(self, x):
        mapped = self.net(x)
        return tfd.Bernoulli(logits=mapped)
    
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

z_size = 100
diag_enc = DiagonalEncoder(z_size=z_size)

joint_enc = JointEncoder(z_size=z_size)
banded_joint_enc = BandedJointEncoder(z_size=z_size)
print("works")

# Read and process the image
# jax = io.imread("./Images/jax.jpg", as_gray=False)  # convert img to greyscale
# jax_gs = color.rgb2gray(jax)
# jax_gs /= 255.0

digit_img = io.imread(".\Images\\2_digit.jpg")

plt.imshow(digit_img, cmap='gray')
plt.title("Image, original, unmodified")
plt.axis('off')
plt.show()

# Encode the images
digit_encoded = diag_enc(digit_img)
print("works after digit_encoded")

plt.imshow(digit_encoded.mean().numpy(), cmap='gray')
plt.title("Encoded image")
plt.axis('off')
plt.show()

"""
Copying over Decoder methods to test on encoded image
"""

class Decoder_2828(tf.keras.Model):
    # Modified base decoder class for MNIST images of size (28,28) instead of HMNIST (64,64)
    def __init__(self, output_size, hidden_sizes=(28,28)):
        """ Decoder parent class with no specified output distribution
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
        """
        super(Decoder_2828, self).__init__()
        self.net = nn_utils.make_nn(output_size, hidden_sizes)

    def __call__(self, x):
        pass

class BernoulliDecoder_2828(Decoder_2828):
    # Modified Bernoulli decoder for (28,28) images
    """ Decoder with Bernoulli output distribution (used for HMNIST) """
    def __call__(self, x):
        mapped = self.net(x)
        return tfd.Bernoulli(logits=mapped)
    
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


bernoulli_decoder = BernoulliDecoder_2828(output_size=28*28)
# gaussian_decoder = GaussianDecoder(output_size=28*28)

# bernoulli_reconstruction = tf.reshape(bernoulli_decoder(jax_scaled_encoded_reshaped).mean(), (180,256))
bernoulli_output = bernoulli_decoder(digit_encoded.mean())
print("Bernoulli output shape:", bernoulli_output.mean().shape)  # Check the shape here

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