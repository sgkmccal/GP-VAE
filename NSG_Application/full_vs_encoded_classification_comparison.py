import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
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


df = pd.read_csv("NSG_Application\\temp_passfail_data.csv")
# print(df.shape)

timestamps = df.iloc[:, 1]
timestamps = pd.to_datetime(timestamps, format='%d.%m.%Y %H:%M')
timestamps = timestamps.view('int64') // 10**9
timestamps = timestamps - timestamps.iloc[0]
print(timestamps)

X = df.iloc[:, 3:58]
X['ScanDateTimeGlasses'] = timestamps
print(X.shape)

print(X.columns)

Y = df.iloc[:, 60]
# print(Y)
# print("Last X col = ", X.iloc[0,53])
# print("X: ", X.shape, "Y: ", Y.shape)

n_train_samples = 600

x_train = X.iloc[:n_train_samples, :].to_numpy().astype('float32')
y_train = Y.iloc[:n_train_samples].to_numpy().astype('float32').reshape(-1,1)

x_test = X.iloc[n_train_samples:, :].to_numpy().astype('float32')
y_test = Y.iloc[n_train_samples:].to_numpy().astype('float32').reshape(-1,1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# n_features = 56 # number of features in X


# Base encoder
class BaseEncoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(BaseEncoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(x_train.shape[1],)),  # Input: x_train features
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(latent_dim, activation='relu')  # Encoded representation
        ])
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')  # binary output - y_train value

    def call(self, inputs):
        encoded = self.encoder(inputs)
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
        self.net = make_cnn(3 * z_size, hidden_sizes, window_size)
        self.data_type = data_type

    def __call__(self, x, training=False):
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)
    
        mapped = self.net(x, training=training)

        # batch_size = mapped.shape[0]
        # time_length = mapped.shape[1]
        batch_size = tf.shape(mapped)[0]  # Dynamic batch size
        time_length = tf.shape(mapped)[1]  # Extract time dimension dynamically

        # Obtain mean and precision matrix components
        num_dim = len(mapped.shape)
        perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        mapped_transposed = tf.transpose(mapped, perm=perm)
        mapped_mean = mapped_transposed[:, :self.z_size]  # Extract mean
        mapped_covar = mapped_transposed[:, self.z_size:]

        # Convert covariance matrix components
        mapped_covar = tf.nn.softplus(mapped_covar)
        mapped_reshaped = tf.reshape(mapped_covar, [batch_size, self.z_size, 2 * time_length])

        dense_shape = [batch_size, self.z_size, time_length, time_length]
        idxs_1 = np.repeat(np.arange(batch_size), self.z_size * (2 * time_length - 1))
        idxs_2 = np.tile(np.repeat(np.arange(self.z_size), (2 * time_length - 1)), batch_size)
        idxs_3 = np.tile(np.concatenate([np.arange(time_length), np.arange(time_length - 1)]), batch_size * self.z_size)
        idxs_4 = np.tile(np.concatenate([np.arange(time_length), np.arange(1, time_length)]), batch_size * self.z_size)
        idxs_all = np.stack([idxs_1, idxs_2, idxs_3, idxs_4], axis=1)

        # Calculate covariance matrix on CPU
        with tf.device('/cpu:0'):
            mapped_values = tf.reshape(mapped_reshaped[:, :, :-1], [-1])
            prec_sparse = tf.sparse.SparseTensor(indices=idxs_all, values=mapped_values, dense_shape=dense_shape)
            prec_sparse = tf.sparse.reorder(prec_sparse)
            prec_tril = tf.sparse_add(tf.zeros(prec_sparse.dense_shape, dtype=tf.float32), prec_sparse)
            eye = tf.eye(num_rows=prec_tril.shape[-1], batch_shape=prec_tril.shape[:-2])
            prec_tril = prec_tril + eye
            cov_tril = tf.linalg.triangular_solve(matrix=prec_tril, rhs=eye, lower=False)
            cov_tril = tf.where(tf.math.is_finite(cov_tril), cov_tril, tf.zeros_like(cov_tril))

        num_dim = len(cov_tril.shape)
        perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        cov_tril_lower = tf.transpose(cov_tril, perm=perm)

        # If training, return mean; otherwise, return full distribution
        if training:
            return mapped_mean  # Return only the mean for training
        else:
            z_dist = tfd.MultivariateNormalTriL(loc=mapped_mean, scale_tril=cov_tril_lower)
            return z_dist  # Return the full distribution for inference



# ---- Base Encoder ----
# encoder training
latent_dims = 5
encoder = BaseEncoder(latent_dims)
encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

encoder.fit(tf.identity(x_train), tf.identity(y_train), epochs=50, batch_size=32, verbose=0)

# Get encoded data
x_train_encoded = encoder(x_train).numpy()
x_test_encoded = encoder(x_test).numpy()

# Define a Gaussian Process Classifier
# kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
kernel = RBF(length_scale = 1.0, length_scale_bounds=(1e-3, 1e4))
gpc_basic = GaussianProcessClassifier(kernel=kernel, random_state=42)
gpc_encoded = GaussianProcessClassifier(kernel=kernel, random_state=42)

# Train GPC on raw data
gpc_basic.fit(x_train, y_train)
y_pred_basic = gpc_basic.predict(x_test)

# Train GPC on encoded data
gpc_encoded.fit(x_train_encoded, y_train)
y_pred_encoded = gpc_encoded.predict(x_test_encoded)
 
# Evaluate and compare performance
acc_basic = accuracy_score(y_test, y_pred_basic)
acc_encoded = accuracy_score(y_test, y_pred_encoded)

print(f"Accuracy (Raw Data): {acc_basic:.4f}")
print(f"Accuracy (Encoded Data): {acc_encoded:.4f}")


# ---- Diagonal Encoder ----
diag_encoder = DiagonalEncoder(latent_dims)
diag_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

# Fit encoder
diag_encoder.fit(tf.identity(x_train), tf.identity(y_train), epochs=50, batch_size=32, verbose=0)

# produce encoded inputs
x_train_encoded_diag = diag_encoder(x_train, training = False).numpy()
x_test_encoded_diag = diag_encoder(x_test, training = False).numpy()

# define GPC for new encoder
gpc_encoded_diag = GaussianProcessClassifier(kernel=kernel, random_state=42)

# train on encoded data
gpc_encoded_diag.fit(x_train_encoded_diag, y_train)

# produce y predictions for encoded x_test
y_pred_encoded_diag = gpc_encoded_diag.predict(x_test_encoded_diag)

# Evaluate and compare performance
# acc_basic = accuracy_score(y_test, y_pred_basic)
acc_encoded_diag = accuracy_score(y_test, y_pred_encoded_diag)

# print(f"Accuracy (Raw Data): {acc_basic:.4f}")
print(f"Accuracy (Diagonal Encoded Data): {acc_encoded_diag:.4f}")


# ---- Joint Encoder ----
joint_encoder = JointEncoder(latent_dims)
joint_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

# Fit encoder
joint_encoder.fit(tf.identity(x_train), tf.identity(y_train), epochs=50, batch_size=32, verbose=0)

# produce encoded inputs
x_train_encoded_joint = joint_encoder(x_train, training = False).numpy()
x_train_encoded_joint = x_train_encoded_joint.reshape(600,latent_dims) # "undoing" addition of time_length term
x_test_encoded_joint = joint_encoder(x_test, training = False).numpy()
x_test_encoded_joint = x_test_encoded_joint.reshape(171,latent_dims) # "undoing" addition of time_length term

# define GPC for new encoder
gpc_encoded_joint = GaussianProcessClassifier(kernel=kernel, random_state=42)

# train on encoded data
gpc_encoded_joint.fit(x_train_encoded_joint, y_train)

# produce y predictions for encoded x_test
y_pred_encoded_joint = gpc_encoded_joint.predict(x_test_encoded_joint)

# Evaluate and compare performance
# acc_basic = accuracy_score(y_test, y_pred_basic)
acc_encoded_joint = accuracy_score(y_test, y_pred_encoded_joint)

# print(f"Accuracy (Raw Data): {acc_basic:.4f}")
print(f"Accuracy (Joint Encoded Data): {acc_encoded_joint:.4f}")


# ---- Joint Encoder ----
bandedjoint_encoder = BandedJointEncoder(latent_dims)
bandedjoint_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

# Fit encoder
bandedjoint_encoder.fit(tf.identity(x_train), tf.identity(y_train), epochs=50, batch_size=32, verbose=0)

# produce encoded inputs
x_train_encoded_bandedjoint = bandedjoint_encoder(x_train, training = False).numpy()
x_train_encoded_bandedjoint = x_train_encoded_bandedjoint.reshape(600,latent_dims) # "undoing" addition of time_length term
x_test_encoded_bandedjoint = bandedjoint_encoder(x_test, training = False).numpy()
x_test_encoded_bandedjoint = x_test_encoded_bandedjoint.reshape(171,latent_dims) # "undoing" addition of time_length term

# define GPC for new encoder
gpc_encoded_bandedjoint = GaussianProcessClassifier(kernel=kernel, random_state=42)

# train on encoded data
gpc_encoded_bandedjoint.fit(x_train_encoded_bandedjoint, y_train)

# produce y predictions for encoded x_test
y_pred_encoded_bandedjoint = gpc_encoded_bandedjoint.predict(x_test_encoded_bandedjoint)

# Evaluate and compare performance
# acc_basic = accuracy_score(y_test, y_pred_basic)
acc_encoded_bandedjoint = accuracy_score(y_test, y_pred_encoded_bandedjoint)

# print(f"Accuracy (Raw Data): {acc_basic:.4f}")
print(f"Accuracy (Banded Joint Encoded Data): {acc_encoded_bandedjoint:.4f}")

