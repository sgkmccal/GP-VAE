import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

def simulate_missingness(dataset, missingness_method, missing_proportion=0.1):
    """
    PARAMETERS:
    dataset: Dataset as .csv file (can expand to .npz files too)
    missingness_method: 'MAR' (Missing at random), 'block' (missingness occurring in blocks, like a failed sensor would cause)
    -----------
    PURPOSE:
    Take complete dataset
    Based on missingness type, set values in dataset to 0
    Return dataset with missing values
    """


    if type(dataset) is not 

    df = pd.read_csv(dataset)

    df = dataset.copy()  # operate on clone of dataset, avoid any overwrite risk
    df_shape = df.shape

    if missingness_method == 'block':

        # Create a mask for structured missingness (block missingness)
        # Start with all True (no missing values)
        mask = np.ones_like(df, dtype=bool)

        for col in range(df_shape[1]):  # For each column
            # Select a random range of rows to mask
            # Number of elements to mask
            num_missing = int(missing_proportion * df_shape[0])
            start_idx = np.random.randint(
                0, df_shape[0] - num_missing + 1)
            mask[start_idx:start_idx + num_missing,
                col] = False  # Mask a vertical section

        # Apply the mask
        df[~mask] = np.nan

        return df

    if missingness_method == 'MAR' or 'mar':

        # create mask with values randomly set to True or False
        mask = np.random.rand(*df_shape) > missing_proportion  

        # create masked version, any [i,j] in mask set to False will make that [i,j]-th element 
        df_masked = df * mask

        return df_masked
    
def feature_select(feature_column_idx_list):
    """
    Pass list of indices (list of integers) or column names (list of strings)
    List is of columns TO BE SELECTED

    e.g. data: ["col1", "col2", "col3", "col4", "col5", "col6", "col7"]
         features: [0, 1, 2]

         RETURNED_DATA: ["col1", "col2", "col3"]
    """

    df_trimmed = pd.DataFrame()
    for val in feature_column_idx_list:
        df_trimmed[] 

def encode_data (data, encoder_type="base", hidden_sizes, kernel_size = 3):
    """
    encoder_type: base, joint, diagonal, 

    Inputs needed by encoder_type:
    Base: data, latent_dim
    Joint: data, z_size, hidden_sizes=(64, 64), window_size=3, transpose=False
    """

    def make_nn(output_size, hidden_sizes):
        """ Creates fully connected neural network
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes. The tuple length sets the number of hidden layers.
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


    class BaseEncoder(tf.keras.Model):
        def __init__(self, latent_dim):
            super(BaseEncoder, self).__init__()
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(data.shape[1],)),  # Input: x_train features
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(latent_dim, activation='relu')  # Encoded representation
            ])
            self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')  # binary output - y_train value

        def call(self, data):
            encoded = self.encoder(data)
            return self.output_layer(encoded)

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
