# Unoptimised kernel methods, mainly for visualising correlations
# SKLearn doesn't have SM implementation (I think)

import numpy as np
import sklearn
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, RationalQuadratic
import matplotlib.pyplot as plt
from scipy.spatial.distance import sqeuclidean 
import math

# set constant σ^2 for all kernels
additive_noise = 1e-6  

# artificial X data
X = np.linspace(0, 10, 100).reshape(-1, 1)

# set constant l for all kernels
length_scale = 1.0  # Set the length scale

import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(X, Y, length_scale=1.0):
    """
    Computes the RBF kernel between two datasets X and Y.

    Parameters:
    - X: array-like, shape (n_samples_X, n_features)
    - Y: array-like, shape (n_samples_Y, n_features)
    - length_scale: float, the length scale parameter

    Returns:
    - Kernel matrix: shape (n_samples_X, n_samples_Y)
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    dists = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * dists / length_scale ** 2)

def matern_kernel(X, Y, length_scale=1.0, nu=1.5):
    """
    Computes the Matérn kernel between two datasets X and Y.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    dists = np.sqrt(np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2))
    if nu == 0.5:
        K = np.exp(-dists / length_scale)
    elif nu == 1.5:
        K = (1 + np.sqrt(3) * dists / length_scale) * np.exp(-np.sqrt(3) * dists / length_scale)
    elif nu == 2.5:
        K = (1 + np.sqrt(5) * dists / length_scale + 5 * dists ** 2 / (3 * length_scale ** 2)) * np.exp(-np.sqrt(5) * dists / length_scale)
    else:
        raise ValueError("Currently, only nu=0.5, 1.5, or 2.5 are supported.")
    return K

def exponential_sine_squared_kernel(X, Y, length_scale=1.0, periodicity=1.0):
    """
    Computes the Exponential-Sine-Squared (Periodic) kernel between two datasets X and Y.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    dists = np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :])
    K = np.exp(-2 * (np.sin(np.pi * dists / periodicity) ** 2) / length_scale ** 2)
    return K

def rational_quadratic_kernel(X, Y, length_scale=1.0, alpha=1.0):
    """
    Computes the Rational Quadratic kernel between two datasets X and Y.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    dists = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2)
    K = (1 + dists / (2 * alpha * length_scale ** 2)) ** -alpha
    return K

def cauchy_kernel(X, Y, length_scale=1.0, nu=1.0):
    """
    Computes the Cauchy kernel between two datasets X and Y.

    Parameters:
    - X: array-like, shape (n_samples_X, n_features)
    - Y: array-like, shape (n_samples_Y, n_features)
    - length_scale: float, the length scale parameter
    - nu: float, the shape parameter (controls tail heaviness)

    Returns:
    - Kernel matrix: shape (n_samples_X, n_samples_Y)
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    dists = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2)
    K = (1 + dists / length_scale ** 2) ** -nu
    return K

# Spectral Mixture kernel - SKLearn has no implementation
def spectralmixture(X, Y, weights, means, scales):
    if Y == None:
        Y = X

    Q, n_features = means.shape
    kernel = np.zeros((X.shape[0], Y.shape[0]))
    
    for q in range(Q):
        w = weights[q]
        mu = means[q]
        sigma = scales[q]
        
        # Pairwise differences
        diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
        dist_sq = np.sum((2 * np.pi * sigma[np.newaxis, np.newaxis, :] * diff) ** 2, axis=2)
        cos_term = np.sum(2 * np.pi * mu[np.newaxis, np.newaxis, :] * diff, axis=2)
        
        kernel += w * np.exp(-0.5 * dist_sq) * np.cos(cos_term)
    
    return kernel


# K_rbf = rbf_kernel(X, X, length_scale=length_scale)
# K_matern_32 = matern_kernel(X, X, length_scale=length_scale, nu=1.5)
# K_matern_52 = matern_kernel(X,X,length_scale=length_scale, nu=2.5)
# K_expsinesqrd = exponential_sine_squared_kernel(X,X,length_scale=length_scale)
# K_rq = rational_quadratic_kernel(X,X, length_scale=length_scale)
# K_cauchy = cauchy_kernel(X,X,length_scale=length_scale, nu=1)

# Plot the kernel matrix as a heatmap
def plot_K(K, kernelname):
    plt.figure(figsize=(8, 6))
    plt.imshow(K, interpolation='nearest', cmap='viridis')
    plt.title(f"{kernelname} Kernel Matrix (Length Scale = {length_scale})")
    plt.colorbar(label="Correlation")
    plt.xlabel("X")
    plt.ylabel("X'")
    plt.show()

# plot_K(K_rbf, "RBF")
# plot_K(K_matern_32, "Matérn 3/2")
# plot_K(K_matern_52, "Matérn 5/2")
# plot_K(K_expsinesqrd, "Exponential Sine Squared")
# plot_K(K_rq, "Rational Quadratic")
# plot_K(K_cauchy, "Cauchy")

ls = [1,2,5,10,20]


for kernel in [rbf_kernel, matern_kernel, exponential_sine_squared_kernel, rational_quadratic_kernel, cauchy_kernel]:
    pass

#rbf
fig, axes = plt.subplots(1, len(ls), figsize=(18,4), constrained_layout = True)
for i, ax in enumerate(axes):
    l = ls[i]
    K = rbf_kernel(X, X, length_scale=l)

    im = ax.imshow(K, interpolation='nearest', cmap='viridis')
    ax.set_title(f"RBF Kernel Matrix (ls = {l})")
    ax.set_xlabel("X")
    ax.set_ylabel("X.T")

cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)
cbar.set_label("Correlation")
plt.show()  

#matern 3/2
fig, axes = plt.subplots(1, len(ls), figsize=(18,4), constrained_layout = True)
for i, ax in enumerate(axes):
    l = ls[i]
    K = matern_kernel(X, X, length_scale=l, nu=1.5)

    im = ax.imshow(K, interpolation='nearest', cmap='viridis')
    ax.set_title(f"Matern 3/2 Kernel Matrix (ls = {l})")
    ax.set_xlabel("X")
    ax.set_ylabel("X.T")

cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)
cbar.set_label("Correlation")
plt.show()  

# matern 5/2
fig, axes = plt.subplots(1, len(ls), figsize=(18,4), constrained_layout = True)
for i, ax in enumerate(axes):
    l = ls[i]
    K = matern_kernel(X, X, length_scale=l, nu=2.5)

    im = ax.imshow(K, interpolation='nearest', cmap='viridis')
    ax.set_title(f"RBF Kernel Matrix (ls = {l})")
    ax.set_xlabel("X")
    ax.set_ylabel("X.T")

cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)
cbar.set_label("Correlation")
plt.show()  

# exp sine sqrd
fig, axes = plt.subplots(1, len(ls), figsize=(18,4), constrained_layout = True)
for i, ax in enumerate(axes):
    l = ls[i]
    K = exponential_sine_squared_kernel(X, X, length_scale=l)

    im = ax.imshow(K, interpolation='nearest', cmap='viridis')
    ax.set_title(f"Exp. Sine Sqrd. Kernel Matrix (ls = {l})")
    ax.set_xlabel("X")
    ax.set_ylabel("X.T")

cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)
cbar.set_label("Correlation")
plt.show()  

# RQ
fig, axes = plt.subplots(1, len(ls), figsize=(18,4), constrained_layout = True)
for i, ax in enumerate(axes):
    l = ls[i]
    K = rational_quadratic_kernel(X, X, length_scale=l)

    im = ax.imshow(K, interpolation='nearest', cmap='viridis')
    ax.set_title(f"Rational Quadratic Kernel Matrix (ls = {l})")
    ax.set_xlabel("X")
    ax.set_ylabel("X.T")

cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)
cbar.set_label("Correlation")
plt.show()  

# Cauchy
fig, axes = plt.subplots(1, len(ls), figsize=(18,4), constrained_layout = True)
for i, ax in enumerate(axes):
    l = ls[i]
    K = cauchy_kernel(X, X, length_scale=l)

    im = ax.imshow(K, interpolation='nearest', cmap='viridis')
    ax.set_title(f"Cauchy Kernel Matrix (ls = {l})")
    ax.set_xlabel("X")
    ax.set_ylabel("X.T")

cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)
cbar.set_label("Correlation")
plt.show()  

