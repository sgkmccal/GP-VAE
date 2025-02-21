import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier as gpc, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, Kernel

df = pd.read_csv("temp_passfail_data.csv")
df = df.iloc[1:, 3:58]
df_shape = df.shape

df_subset = df.iloc[:60, :]
df_subset_shape = df_subset.shape

np.random.seed(1234)

scaler = StandardScaler()
df = scaler.fit_transform(df)
df = pd.DataFrame(df)

# Full dataset
df_full = np.load("FullDataset.npz")
# print(df_full)

X_full = df_full["array1"]
scaler2 = StandardScaler()
X_full = scaler2.fit_transform(X_full)

# DATA MISSING IN BLOCKS, SENSOR FAILURE ETC.
X_block_missing = X_full.copy()

# Create a mask for structured missingness (block missingness)
missing_fraction = 0.7  # Proportion of elements to remove per column
# Start with all True (no missing values)
mask = np.ones_like(X_block_missing, dtype=bool)

for col in range(X_block_missing.shape[1]):  # For each column
    # Select a random range of rows to mask
    # Number of elements to mask
    num_missing = int(missing_fraction * X_block_missing.shape[0])
    start_idx = np.random.randint(
        0, X_block_missing.shape[0] - num_missing + 1)
    mask[start_idx:start_idx + num_missing,
         col] = False  # Mask a vertical section

# Apply the mask
X_block_missing_masked = X_block_missing.copy()
X_block_missing_masked[~mask] = np.nan

X_block_missing_masked = pd.DataFrame(X_block_missing_masked)

# Perform linear interpolation and fill missing values
X_block_missing_masked_lerp = X_block_missing_masked.interpolate(
    method='linear', axis=1).fillna(method='bfill').fillna(method='ffill')
X_block_missing_masked_lerp = np.asarray(X_block_missing_masked_lerp)

# Calculate MAE and RMSE only for imputed elements
imputed_mask = ~mask  # This identifies the elements that were imputed
imputed_values = X_block_missing_masked_lerp[imputed_mask]
true_values = X_full[imputed_mask]  # Get true values from the original dataset

x_block_missing_mae = mean_absolute_error(true_values, imputed_values)
# print("MAE for imputed values: ", x_block_missing_mae)

x_block_missing_rmse = math.sqrt(
    mean_squared_error(true_values, imputed_values))
# print("RMSE for imputed values: ", x_block_missing_rmse)


####################

def gp_impute(data, kernel, missing_value=np.nan, true_data=None):
    """
    Impute missing values in a 2D array using Gaussian Processes.

    Parameters:
    - data: 2D NumPy array or pandas DataFrame with missing values.
    - kernel: Kernel to use for Gaussian Process.
    - missing_value: Value representing missing data (default: np.nan).
    - true_data: Original dataset to compare for metrics (optional).

    Returns:
    - imputed_data: Data with missing values imputed.
    - mae: Mean Absolute Error for imputed values (if `true_data` is provided).
    - rmse: Root Mean Squared Error for imputed values (if `true_data` is provided).
    """
    model_kernel = kernel

    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        data_np = data.values
    else:
        data_np = data

    imputed_data = data_np.copy()
    rows, cols = imputed_data.shape

    missing_mask = np.isnan(data_np) if np.isnan(
        missing_value) else (data_np == missing_value)

    for col in range(cols):
        # Identify observed and missing indices for the column
        observed_indices = ~missing_mask[:, col]
        missing_indices = missing_mask[:, col]

        if np.any(missing_indices):
            # Use row indices as the independent variable for GP
            X_observed = np.where(observed_indices)[
                0].reshape(-1, 1)  # Indices of observed rows
            # Observed values
            y_observed = imputed_data[observed_indices, col]

            X_missing = np.where(missing_indices)[
                0].reshape(-1, 1)   # Indices of missing rows

            # Train GP on observed data
            gp = GaussianProcessRegressor(kernel=model_kernel, random_state=42)
            gp.fit(X_observed, y_observed)

            # Predict missing values
            y_missing_pred, _ = gp.predict(X_missing, return_std=True)

            # Fill in the missing values
            imputed_data[missing_indices, col] = y_missing_pred

    # If true_data is provided, calculate MAE and RMSE for imputed values
    if true_data is not None:
        true_data_np = true_data.values if isinstance(
            true_data, pd.DataFrame) else true_data
        imputed_values = imputed_data[missing_mask]
        true_values = true_data_np[missing_mask]

        mae = mean_absolute_error(true_values, imputed_values)
        rmse = math.sqrt(mean_squared_error(true_values, imputed_values))

        return imputed_data, mae, rmse

    return imputed_data


# Load full dataset
df_full = np.load("FullDataset.npz")
# print(df_full)

X_full = df_full["array1"]
# print(X_full)
#Y_full = df_full["array2"]

scaler2 = StandardScaler()

X_full = scaler2.fit_transform(X_full)


np.random.seed(42)
full_mask = np.random.rand(*X_full.shape) > 0.2
X_full_masked = X_full.copy()
X_full_masked[full_mask] = np.nan

print("Xfullmasked shape = ", X_full_masked.shape)
print("full_mask shape = ", full_mask.shape)
print(np.isnan(X_full_masked).sum())


class SpectralMixtureKernel(Kernel):
    """
    Spectral Mixture Kernel for use with sklearn's GaussianProcessRegressor.
    """

    def __init__(self, num_mixtures=1, weights=None, means=None, variances=None):
        self.num_mixtures = num_mixtures
        self.weights = weights
        self.means = means
        self.variances = variances

        # Hyperparameters for weights, means, and variances
        self.weight_hyperparameter = Hyperparameter(
            "weights", "fixed", (1e-6, 1e2), n_elements=num_mixtures
        )
        self.mean_hyperparameter = Hyperparameter(
            "means", "fixed", (1e-6, 1e2), n_elements=num_mixtures
        )
        self.variance_hyperparameter = Hyperparameter(
            "variances", "fixed", (1e-6, 1e2), n_elements=num_mixtures
        )

    def _initialize_parameters(self, X):
        """
        Initialize weights, means, and variances if not provided.
        """
        if self.weights is None:
            self.weights = np.random.uniform(0.1, 1.0, self.num_mixtures)
        if self.means is None:
            self.means = np.random.uniform(
                0.1, 1.0, (self.num_mixtures, X.shape[1]))
        if self.variances is None:
            self.variances = np.random.uniform(
                0.1, 1.0, (self.num_mixtures, X.shape[1]))

    def __call__(self, X, Y=None, eval_gradient=False):
        """
        Compute the kernel matrix between inputs X and Y.
        """
        if Y is None:
            Y = X

        self._initialize_parameters(X)

        N, D = X.shape
        M, _ = Y.shape
        K = np.zeros((N, M))

        for q in range(self.num_mixtures):
            weight = self.weights[q]
            mean = self.means[q]
            variance = self.variances[q]

            # Compute pairwise squared distances
            diff = X[:, None, :] - Y[None, :, :]  # Shape: (N, M, D)
            # Weighted distance, Shape: (N, M)
            dist2 = np.sum((diff**2) * variance, axis=2)

            # Exponential term
            exp_term = np.exp(-2 * np.pi**2 * dist2)

            # Cosine term
            cos_term = np.cos(2 * np.pi * np.sum(diff * mean, axis=2))

            # Weighted sum
            K += weight * exp_term * cos_term

        if eval_gradient:
            raise NotImplementedError(
                "Gradient computation is not implemented for this kernel.")

        return K

    def diag(self, X):
        """
        Compute the diagonal of the kernel matrix.
        """
        self._initialize_parameters(X)
        return np.sum(self.weights)

    def is_stationary(self):
        """
        Whether the kernel is stationary.
        """
        return True

# kernel = RBF(length_scale=1.0)  # Example kernel


for i in range(1, 20):
    kernel = SpectralMixtureKernel(num_mixtures=i)
    imputed_data, mae, rmse = gp_impute(
        X_full_masked, kernel=kernel, true_data=X_full)

    print(f"Num. mixture components: {i}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
