# Compare performance of different kernels on full dataset,
# Later on reconstructed and encoded data

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Kernel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import sklearn.gaussian_process.kernels as kernels
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist

#%%
# df = pd.read_csv("NSG_Application\\temp_passfail_data.csv")
# # print(df.shape)

# timestamps = df.iloc[:, 1]
# timestamps = pd.to_datetime(timestamps, format='%d.%m.%Y %H:%M')
# timestamps = timestamps.view('int64') // 10**9
# timestamps = timestamps - timestamps.iloc[0]

# X = df.iloc[:, 3:58]
# X['ScanDateTimeGlasses'] = timestamps
# Y = df.iloc[:, 60]

# n_train_samples = 600

# x_train = X.iloc[:n_train_samples, :].to_numpy().astype('float32')
# y_train = Y.iloc[:n_train_samples].to_numpy().astype('float32').reshape(-1,1)

# x_test = X.iloc[n_train_samples:, :].to_numpy().astype('float32')
# y_test = Y.iloc[n_train_samples:].to_numpy().astype('float32').reshape(-1,1)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#%%
np.random.seed(1337)

class spectralmixture(Kernel):
    """
    Computes the spectral mixture kernel.

    Parameters:
    - X: Input data of shape (n_samples, n_features) or (n_samples,) for single feature.
    - weights: Array of shape (Q,) representing weights for each mixture component.
    - means: Array of shape (Q, n_features) for mean frequencies.
    - scales: Array of shape (Q, n_features) for length scales.
    - Q: Number of spectral mixture components.

    Returns:
    - kernel: Kernel matrix of shape (n_samples, n_samples).
    """
    # Ensure X is 2D

    def __init__(self):
        pass

    def __call__(self):
        pass

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape

    # Set default weights if not provided
    if weights is None:
        weights = np.ones(Q) / Q
    else:
        assert len(weights) == Q, "Weights must have length equal to Q."

    # Set default means if not provided
    if means is None:
        means = np.zeros((Q, n_features))
    else:
        assert means.shape == (
            Q, n_features), "Means must have shape (Q, n_features)."

    # Set default scales if not provided
    if scales is None:
        scales = np.ones((Q, n_features))
    else:
        assert scales.shape == (
            Q, n_features), "Scales must have shape (Q, n_features)."

    # Initialize kernel matrix
    kernel = np.zeros((n_samples, n_samples))

    for q in range(Q):
        w = weights[q]
        mu = means[q]
        sigma = scales[q]

        # Pairwise differences
        # Shape: (n_samples, n_samples)
        pairwise_dists = cdist(X, X, 'euclidean')

        # Compute squared distance term
        dist_sq = np.sum(
            (2 * np.pi * sigma * pairwise_dists[..., np.newaxis]) ** 2,
            axis=-1
        )

        # Compute cosine term
        cos_term = np.sum(
            2 * np.pi * mu * pairwise_dists[..., np.newaxis],
            axis=-1
        )

        # Add contribution from this component
        kernel += w * np.exp(-0.5 * dist_sq) * np.cos(cos_term)

    #return kernel


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
    K = spectralmixture(X, weights, means, scales, Q) + \
        ((noise**2) * np.eye(len(X)))

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


def train_spectral_mixture(X, y, Q=5, noise=1e-2):
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

    # Initialize parameters
    weights = np.ones(Q) / Q
    means = np.random.rand(Q, n_features)
    scales = np.random.rand(Q, n_features)

    params = np.concatenate([weights, means.flatten(), scales.flatten()])

    # Define bounds for optimization
    bounds = [(1e-5, 1e3)] * len(weights) + [(1e-5, 1e3)] * \
        means.size + [(1e-5, 1e3)] * scales.size

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
    opt_scales = res.x[Q + Q * n_features:].reshape(Q, n_features)

    return opt_weights, opt_means, opt_scales


# x = np.linspace(0, 10, 100)
# x2 = np.linspace(0, 5, 100)
# x_arr = np.hstack((x, x2))

x_arr = np.random.rand(100, 3)

y = np.sum(np.sin(x_arr) + np.cos(x_arr), axis=1)

Q = 2
K = spectralmixture(x_arr, Q=Q)
w_opt, m_opt, s_opt = train_spectral_mixture(x_arr, y, Q=Q)
K_opt = spectralmixture(x_arr, w_opt, m_opt, s_opt, Q=Q)

plt.imshow(K, cmap='viridis')
plt.title("Unoptimised K")
plt.show()

plt.imshow(K_opt, cmap='viridis')
plt.title("Optimised K")
plt.show()
