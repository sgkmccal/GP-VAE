import numpy as np
from sklearn.gaussian_process.kernels import Kernel
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class SpectralMixtureKernel(Kernel):
    """
    Spectral Mixture Kernel for use with sklearn's GaussianProcessRegressor.
    """
    def __init__(self, Q=2, weights=None, means=None, variances=None):
        """
        Initialize the Spectral Mixture Kernel.

        Parameters:
        - Q: Number of mixture components (Q).
        - weights: Array of weights for the components (default: random initialization).
        - means: Array of mean frequencies for the components (default: random initialization).
        - variances: Array of variances (inverse length scales squared) (default: random initialization).
        """
        self.Q = Q
        self.weights = weights
        self.means = means
        self.variances = variances

    def _initialize_parameters(self, X):
        """
        Initialize weights, means, and variances if not provided.
        """
        if self.weights is None:
            self.weights = np.ones(self.Q) / self.Q  # Uniform weights
        if self.means is None:
            self.means = np.random.uniform(0, 1, (self.Q, X.shape[1]))
        if self.variances is None:
            self.variances = np.random.uniform(1e-2, 1, (self.Q, X.shape[1]))

    def __call__(self, X, Y=None, eval_gradient=False):
        """
        Compute the kernel matrix between inputs X and Y.

        Parameters:
        - X: Input array of shape (N, D).
        - Y: Input array of shape (M, D). If None, use X.
        - eval_gradient: Whether to compute the gradient (not implemented here).

        Returns:
        - Kernel matrix of shape (N, M).
        """
        if Y is None:
            Y = X

        self._initialize_parameters(X)

        N, D = X.shape
        M, _ = Y.shape
        K = np.zeros((N, M))

        for q in range(self.Q):
            weight = self.weights[q]
            mean = self.means[q]
            variance = self.variances[q]

            # Compute pairwise squared distances
            diff = X[:, None, :] - Y[None, :, :]  # Shape: (N, M, D)
            dist2 = np.sum((diff**2) * variance, axis=2)  # Weighted distance, Shape: (N, M)

            # Exponential term
            exp_term = np.exp(-2 * np.pi**2 * dist2)

            # Cosine term
            cos_term = np.cos(2 * np.pi * np.sum(diff * mean, axis=2))

            # Weighted sum
            K += weight * exp_term * cos_term

        if eval_gradient:
            # Gradient computation is not implemented in this example
            raise NotImplementedError("Gradient computation is not implemented for this kernel.")
        
        return K

    def diag(self, X):
        """
        Compute the diagonal of the kernel matrix.

        Parameters:
        - X: Input array of shape (N, D).

        Returns:
        - Diagonal of the kernel matrix, shape (N,).
        """
        self._initialize_parameters(X)
        return np.sum(self.weights)

    def is_stationary(self):
        """
        Whether the kernel is stationary.
        """
        return True

def train_spectral_mixture(X, y, Q=2, noise=1e-3):
    """
    Train the spectral mixture kernel by maximizing the log marginal likelihood.

    Args:
        X (ndarray): Input data of shape (N, D) or (N,).
        y (ndarray): Target values of shape (N,).
        Q (int): Number of spectral mixture components.
        noise (float): Noise level for the Gaussian Process.

    Returns:
        tuple: Optimized weights, means, and scales.
    """
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape

    weights = np.ones(Q) / Q  # Uniform weights
    means = np.random.uniform(0, 1, (Q, X.shape[1]))
    variances = np.random.uniform(1e-2, 1, (Q, X.shape[1]))

    params = np.concatenate([weights, means.flatten(), variances.flatten()])

    # Define bounds for optimization
    bounds = [(1e-5, 1e3)] * len(weights) + [(1e-5, 1e3)] * \
        means.size + [(1e-5, 1e3)] * variances.size
    
    def log_marginal_likelihood(params, X, y, Q, D, noise):
        """
        Computes the log marginal likelihood for the spectral mixture kernel for n-dimensional data.

        Args:
            params (ndarray): Flattened kernel parameters (weights, means, scales).
            X (ndarray): Input data of shape (N, D).
            y (ndarray): Target values of shape (N,).
            Q (int): Number of spectral components.
            D (int): Number of features.
            noise (float): Noise level for the Gaussian Process.

        Returns:
            float: Log marginal likelihood (scalar).
        """
        # Extract kernel parameters
        weights = params[:Q]
        means = params[Q:Q + Q * D].reshape(Q, D)
        scales = params[Q + Q * D:].reshape(Q, D)

        # Compute covariance matrix
        kernel = SpectralMixtureKernel(Q=Q, weights=weights, means=means, variances=scales)
        K = kernel(X) + ((noise**2) * np.eye(len(X)))

        # Cholesky decomposition
        L = np.linalg.cholesky(K)

        # Solve for alpha (K^{-1} y)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

        # Log determinant of K
        log_det = 2 * np.sum(np.log(np.diag(L)))

        # Compute log marginal likelihood
        lml = -0.5 * y.T @ alpha - 0.5 * log_det - 0.5 * len(y) * np.log(2 * np.pi)

        # print("lml = ", lml)

        return float(lml)  # Ensure scalar output

    # Optimize the parameters
    res = minimize(
        log_marginal_likelihood,
        params,
        args=(X, y, Q, n_features, noise),
        method="L-BFGS-B",
        bounds=bounds
    )

    # Extract optimized parameters
    opt_weights = res.x[:Q]
    opt_means = res.x[Q:Q + Q * n_features].reshape(Q, n_features)
    opt_variances = res.x[Q + Q * n_features:].reshape(Q, n_features)

    return opt_weights, opt_means, opt_variances


# df_full = np.load("NSG_Application\\FullDataset.npz")
# X = df_full["array1"]
# Y = df_full["array_2"]
# n_train_samples = 600

# x_train = X[:n_train_samples, :]
# y_train = Y[:n_train_samples].reshape(-1,1)

# x_test = X[n_train_samples:, :]
# y_test = Y[n_train_samples:]

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


# for i in range(1, 20):
#     sm_kernel = SpectralMixtureKernel(Q = i)

#     gpc = GaussianProcessClassifier(kernel=sm_kernel)
#     gpc.fit(x_train, y_train)
#     print(f"GP w SM kernel, {i} components, score: ", gpc.score(x_test, y_test))

n_train = 400
x = np.linspace(0,10,500)
x1 = np.sin(x)
x2 = np.sin(x) + 0.1
X = np.vstack((x1,x2)).T

x_train = X[:n_train, :]
x_test = X[n_train:, :]

y = np.sin(np.sum(X, axis=1))
y_train = y[:n_train]
y_test = y[n_train:]

smk = SpectralMixtureKernel()
K_prior = smk(x_train)
# w_opt, m_opt, s_opt = train_spectral_mixture()

K_test_test = smk(x_test, x_test)
K_train_test = smk(x_train, x_test)

mu_post = K_train_test.T @ np.linalg.inv(K_prior + np.eye(len(x_train))*1e-6) @ y_train
K_post = K_test_test - K_train_test.T @ np.linalg.inv(K_prior + np.eye(len(x_train))*1e-6) @ K_train_test

# plt.imshow(K_prior, cmap='viridis')
# plt.title("Prior K")
# plt.show()

# plt.imshow(K_post, cmap='viridis')
# plt.title("Posterior K")
# plt.show()

w_opt, m_opt, v_opt = train_spectral_mixture(X=x_train, y=y_train)

smk_opt = SpectralMixtureKernel(weights=w_opt, means=m_opt, variances=v_opt)
K_opt_prior = smk_opt(X=x_train)

plt.imshow(K_opt_prior, cmap='viridis')
plt.title("Prior K w/ optimised w,m,s")
plt.show()

K_opt_test_test = smk_opt(x_test)
K_opt_train_test = smk_opt(x_train, x_test)
K_opt_post = K_opt_test_test - K_opt_train_test.T @ np.linalg.inv(K_opt_prior * np.eye(len(x_train)) * 1e-6) @ K_opt_train_test

plt.imshow(K_opt_post, cmap='viridis')
plt.title("K posterior w/ opt hyperparams")
plt.show()