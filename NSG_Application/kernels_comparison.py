# Compare performance of different kernels on full dataset, 
# Later on reconstructed and encoded data

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, RationalQuadratic, ExpSineSquared
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

df = pd.read_csv("NSG_Application\\temp_passfail_data.csv")
# print(df.shape)

timestamps = df.iloc[:, 1]
timestamps = pd.to_datetime(timestamps, format='%d.%m.%Y %H:%M')
timestamps = timestamps.view('int64') // 10**9
timestamps = timestamps - timestamps.iloc[0]

X = df.iloc[:, 3:58]
X['ScanDateTimeGlasses'] = timestamps
Y = df.iloc[:, 60]

n_train_samples = 600

x_train = X.iloc[:n_train_samples, :].to_numpy().astype('float32')
y_train = Y.iloc[:n_train_samples].to_numpy().astype('float32').reshape(-1,1)

x_test = X.iloc[n_train_samples:, :].to_numpy().astype('float32')
y_test = Y.iloc[n_train_samples:].to_numpy().astype('float32').reshape(-1,1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


def spectralmixture(X, Y, weights, means, scales):
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

def log_marginal_likelihood(params, X, y, Q, n_features, noise):
    weights = params[:Q]
    means = params[Q:Q + Q * n_features].reshape(Q, n_features)
    scales = params[Q + Q * n_features:].reshape(Q, n_features)
    
    K = spectralmixture(X, X, weights, means, scales) + noise**2 * np.eye(len(X))
    L = np.linalg.cholesky(K)
    
    # Compute log marginal likelihood
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    log_det = 2 * np.sum(np.log(np.diag(L)))
    nll = 0.5 * y.T @ alpha + 0.5 * log_det + 0.5 * len(y) * np.log(2 * np.pi)
    return nll

def train_spectral_mixture(X, y, Q=2, noise=1e-2):
    n_features = X.shape[1]
    
    # Initial parameters
    weights = np.ones(Q) / Q
    means = np.random.rand(Q, n_features)
    scales = np.random.rand(Q, n_features)
    
    params = np.concatenate([weights, means.flatten(), scales.flatten()])
    
    # Optimize the parameters
    res = minimize(
        log_marginal_likelihood,
        params,
        args=(X, y, Q, n_features, noise),
        method="L-BFGS-B",
        bounds=[(1e-5, 10)] * len(params)
    )
    
    # Extract optimized parameters
    opt_weights = res.x[:Q]
    opt_means = res.x[Q:Q + Q * n_features].reshape(Q, n_features)
    opt_scales = res.x[Q + Q * n_features:].reshape(Q, n_features)
    
    return opt_weights, opt_means, opt_scales

# Example Usage
np.random.seed(42)
X = np.linspace(0, 10, 20).reshape(-1, 1)
y = np.sin(X).flatten() + 0.1 * np.random.randn(20)

opt_weights, opt_means, opt_scales = train_spectral_mixture(X, y, Q=2)

print("Optimized Weights:", opt_weights)
print("Optimized Means:", opt_means)
print("Optimized Scales:", opt_scales)

#sync