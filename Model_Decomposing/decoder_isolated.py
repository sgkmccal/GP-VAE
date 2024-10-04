# Decoders
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from lib import nn_utils

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
    
# basic_decoder = Decoder()
output_size = 2
bernoulli_decoder = BernoulliDecoder(output_size=output_size)
gaussian_decoder = GaussianDecoder(output_size=output_size)
print("worked")