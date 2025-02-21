# Compare performance of different kernels on full dataset,
# Later on reconstructed and encoded data

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, RationalQuadratic, ExpSineSquared
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import sklearn.gaussian_process.kernels as kernels
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist

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


np.random.seed(1234)


# def spectralmixture(X_train, X_test, y_train,
#                     weights=None, means=None, variances=None, Q=2,
#                     plot_prior_K=False, plot_posterior_K=True,
#                     calculate_posterior=True):
# """
# Computes the spectral mixture kernel.

# Parameters:
# - X: Input data of shape (n_samples, n_features) or (n_samples,) for single feature.
# - weights: Array of shape (Q,) representing weights for each mixture component.
# - means: Array of shape (Q, n_features) for mean frequencies.
# - scales: Array of shape (Q, n_features) for length scales.
# - Q: Number of spectral mixture components.

# Returns:
# - kernel: Kernel matrix of shape (n_samples, n_samples).
# """

# def calc_K_prior(X, weights, means, variances):
#     # Ensure X is 2D
#     if X.ndim == 1:
#         X = X.reshape(-1, 1)

#     n_samples, n_features = X.shape

#     # Set default weights if not provided
#     if weights is None:
#         weights = np.ones(Q) / Q
#     else:
#         assert len(weights) == Q, "Weights must have length equal to Q."

#     # Set default means if not provided
#     if means is None:
#         means = np.zeros((Q, n_features))
#     else:
#         assert means.shape == (
#             Q, n_features), "Means must have shape (Q, n_features)."

#     # Set default scales if not provided
#     if variances is None:
#         variances = np.ones((Q, n_features))
#     else:
#         assert variances.shape == (
#             Q, n_features), "Scales must have shape (Q, n_features)."

#     # Initialize kernel matrix
#     kernel = np.zeros((n_samples, n_samples))

#     for q in range(Q):
#         w = weights[q]
#         mu = means[q]
#         sigma = variances[q]

#         # Pairwise differences
#         # Shape: (n_samples, n_samples)
#         pairwise_dists = cdist(X, X, 'euclidean')

#         # Compute squared distance term
#         dist_sq = np.sum(
#             (2 * np.pi * sigma * pairwise_dists[..., np.newaxis]) ** 2,
#             axis=-1
#         )

#         # Compute cosine term
#         cos_term = np.sum(
#             2 * np.pi * mu * pairwise_dists[..., np.newaxis],
#             axis=-1
#         )

#         # Add contribution from this component
#         kernel += w * np.exp(-0.5 * dist_sq) * np.cos(cos_term)

#     return kernel

# def calc_K_train_test(self, X_train, X_test):
#     K = self.calc_K_prior(X_train, self.weights, self.means, self.variances)
#     return K

# def calc_K_test_test():

#     # return K
#     pass

# def calc_posterior_mean():

#     # return mu
#     pass

# def calc_posterior_covmat():

#     # return K
#     pass

# def cholesky():

#     # return decomp
#     pass

# K_prior = calc_K_prior(X_train)
# print("weights after calc K prior ", weights)


# def log_marginal_likelihood(params, X, y, Q, D, noise):
#     """
#     Computes the log marginal likelihood for the spectral mixture kernel for n-dimensional data.

#     Args:
#         params (ndarray): Flattened kernel parameters (weights, means, scales).
#         X (ndarray): Input data of shape (N, D).
#         y (ndarray): Target values of shape (N,).
#         Q (int): Number of spectral components.
#         D (int): Number of features.
#         noise (float): Noise level for the Gaussian Process.

#     Returns:
#         float: Log marginal likelihood (scalar).
#     """
#     # Extract kernel parameters
#     weights = params[:Q]
#     means = params[Q:Q + Q * D].reshape(Q, D)
#     scales = params[Q + Q * D:].reshape(Q, D)

#     # Compute covariance matrix
#     K = spectralmixture(X, weights, means, scales, Q) + \
#         ((noise**2) * np.eye(len(X)))

#     # Cholesky decomposition
#     L = np.linalg.cholesky(K)

#     # Solve for alpha (K^{-1} y)
#     alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

#     # Log determinant of K
#     log_det = 2 * np.sum(np.log(np.diag(L)))

#     # Compute log marginal likelihood
#     lml = -0.5 * y.T @ alpha - 0.5 * log_det - 0.5 * len(y) * np.log(2 * np.pi)

#     # print("lml = ", lml)

#     return float(lml)  # Ensure scalar output


# def train_spectral_mixture(X, y, Q=5, noise=1e-2):
#     """
#     Train the spectral mixture kernel by maximizing the log marginal likelihood.

#     Args:
#         X (ndarray): Input data of shape (N, D) or (N,).
#         y (ndarray): Target values of shape (N,).
#         Q (int): Number of spectral mixture components.
#         noise (float): Noise level for the Gaussian Process.

#     Returns:
#         tuple: Optimized weights, means, and scales.
#     """
#     # Ensure X is 2D
#     if X.ndim == 1:
#         X = X.reshape(-1, 1)

#     n_samples, n_features = X.shape

#     # Initialize parameters
#     weights = prior_weights(Q)
#     means = prior_means(Q, n_features)
#     scales = prior_scales(Q, n_features)

#     params = np.concatenate([weights, means.flatten(), scales.flatten()])

#     # Define bounds for optimization
#     bounds = [(1e-5, 1e3)] * len(weights) + [(1e-5, 1e3)] * \
#         means.size + [(1e-5, 1e3)] * scales.size

#     # Optimize the parameters
#     res = minimize(
#         log_marginal_likelihood,
#         params,
#         args=(X, y, Q, n_features, noise),
#         method="L-BFGS-B",
#         bounds=bounds
#     )

#     # Extract optimized parameters
#     opt_weights = res.x[:Q]
#     opt_means = res.x[Q:Q + Q * n_features].reshape(Q, n_features)
#     opt_scales = res.x[Q + Q * n_features:].reshape(Q, n_features)

#     return opt_weights, opt_means, opt_scales


def prior_means(Q, n_features):
    means = np.random.rand(Q, n_features)
    return means


def prior_variances(Q, n_features):
    variances = np.random.rand(Q, n_features)
    return variances


def prior_weights(Q):
    weights = np.ones(Q) / Q
    return weights


def calc_K(Xa, Xb=None, weights=None, means=None, variances=None, Q=2):
    # Ensure X is 2D
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)

    if Xb is None:
        Xb = Xa
    else:
        if Xb.ndim == 1:
            Xb = Xb.reshape(-1, 1)

    n_samples_a, n_features = Xa.shape
    n_samples_b = Xb.shape[0]

    # Set default weights if not provided
    if weights is None:
        weights = prior_weights(Q)

    # Set default means if not provided
    if means is None:
        means = prior_means(Q, n_features)

    # Set default scales if not provided
    if variances is None:
        variances = prior_variances(Q, n_features)

    # Initialize kernel matrix
    kernel = np.zeros((n_samples_a, n_samples_b))

    for q in range(Q):
        w = weights[q]
        mu = means[q]
        sigma = variances[q]

        # Pairwise differences
        # Shape: (n_samples, n_samples)
        pairwise_dists = cdist(Xa, Xb, 'euclidean')

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

    return kernel, weights, means, variances


def calc_K_post(X_train, X_test, y_train, K_prior,
                weights, means, variances,
                Q, noise=1e-2):

    weights = weights
    means = means
    variances = variances

    if K_prior.all() == None:
        K_prior, weights, means, variances = calc_K(
            X_train, X_train, weights, means, variances, Q)

    K_test_test = calc_K(X_test, X_test, weights, means, variances, Q)[0]
    K_train_test = calc_K(X_train, X_test, weights, means, variances, Q)[0]
    print("Kprior: ", K_prior)
    print("K**: ", K_test_test)
    print("Kt*: ", K_train_test)

    K_post = K_test_test - \
        K_train_test.T @  \
        np.linalg.inv(K_prior + np.eye(len(K_prior))*noise) @  \
        y_train

    return K_post


def covariance_to_frequencies(cov_matrix):
    """
    Converts a covariance matrix to its spectral representation (frequencies).

    Args:
        cov_matrix (ndarray): Covariance matrix of shape (n, n).

    Returns:
        tuple: Eigenvalues and eigenvectors of the covariance matrix.
    """
    # Ensure the matrix is symmetric positive semi-definite
    assert cov_matrix.shape[0] == cov_matrix.shape[1], "Covariance matrix must be square."

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvalues, eigenvectors


def plot_frequencies(cov_matrix):
    """
    Plots the frequencies based on the spectral decomposition of the covariance matrix.

    Args:
        cov_matrix (ndarray): Covariance matrix.
    """
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Plot the eigenvalues (power at each frequency)
    plt.figure(figsize=(10, 6))
    plt.plot(eigenvalues, marker='o')
    plt.title("Frequencies from Covariance Matrix")
    plt.xlabel("Component Index")
    plt.ylabel("Eigenvalue (Power)")
    plt.grid(True)
    plt.show()

    # Optionally plot a few eigenvectors
    plt.figure(figsize=(10, 6))
    for i in range(min(5, eigenvectors.shape[1])):  # Plot up to 5 eigenvectors
        plt.plot(eigenvectors[:, i], label=f"Eigenvector {i+1}")
    plt.title("Eigenvectors (Basis Functions)")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()


n = 500

x = np.linspace(0, 10, n)
z = np.linspace(4, 6, n)
x1 = np.sin(x)
x2 = np.cos(x)
z1 = np.sin(z)
z2 = np.cos(z)

x_arr = np.vstack((x1, x2)).T
z_arr = np.vstack((z1, z2)).T
# xz_arr = np.hstack((x_arr, z_arr)).T
xz_arr = np.vstack((x, z))

y = np.sum(xz_arr, axis=0)

n_train = int(n*0.8)

x_train = x_arr[:n_train, :]
x_test = x_arr[n_train:, :]

y_train = y[:n_train]
y_test = y[n_train:]

Q = 2
noise = 1e-3

K_prior, w, m, v = calc_K(x_train, x_train, Q=Q)
K_post = calc_K_post(x_train, x_test, y_train, K_prior, w, m, v, Q)


plt.plot(x_arr)
plt.show()

plt.imshow(K_post, cmap='viridis')
plt.show()

sf = covariance_to_frequencies(K_post)
sf_eigenvalues = sf[0]
sf_eigenvectors = sf[1]
# plot_frequencies(K_post)

plt.plot(sf_eigenvalues)
plt.title("Eigenvalues - Power Distribution")
plt.show()

plt.imshow(sf_eigenvectors, cmap='viridis')
plt.title("Eigenvectors - Basis Functions")
plt.show()

# w_opt, m_opt, s_opt = train_spectral_mixture(x_train, y_train, Q=Q)
# K_prior_opt = spectralmixture(x_train, w_opt, m_opt, s_opt, Q=Q)

# plt.imshow(K_prior, cmap='viridis')
# plt.title("Unoptimised K_prior")
# plt.show()

# plt.imshow(K_prior_opt, cmap='viridis')
# plt.title("Optimised K_prior")
# plt.show()

# k_train_test = spectralmixture(x_train, x_test)  # k(x,x*)
# k_test = spectralmixture(x_test, x_test)  # k(x*, x*)
# K_posterior = k_test - k_train_test.T @ np.linalg.inv(K_prior) @ k_train_test
