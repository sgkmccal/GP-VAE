from lib import models
from models import HI_VAE, rbf
import tensorflow as tf


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

gpvae = GP_VAE()
print("works")