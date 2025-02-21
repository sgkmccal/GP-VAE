import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

df = pd.read_csv("NSG_Application\\temp_passfail_data.csv")
n_train_samples = 600
X = np.asarray(df.iloc[:, 3:58])
Y = np.asarray(df.iloc[:, 60])

scaler = StandardScaler()
X = scaler.fit_transform(X)

def gp_impute(data, kernel, missing_value=np.nan):
    """
    Impute missing values in a 2D array using Gaussian Processes.

    Parameters:
    - data: 2D NumPy array or pandas DataFrame with missing values.
    - missing_value: Value representing missing data (default: np.nan).

    Returns:
    - imputed_data: Data with missing values imputed.
    """
    model_kernel = kernel

    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        data_np = data.values
    else:
        data_np = data

    imputed_data = data_np.copy()
    rows, cols = imputed_data.shape

    for col in range(cols):
        # Identify observed and missing indices for the column
        observed_indices = ~np.isnan(imputed_data[:, col])
        missing_indices = np.isnan(imputed_data[:, col])

        if np.any(missing_indices):
            # Use row indices as the independent variable for GP
            X_observed = np.where(observed_indices)[0].reshape(-1, 1)  # Indices of observed rows
            y_observed = imputed_data[observed_indices, col]           # Observed values

            X_missing = np.where(missing_indices)[0].reshape(-1, 1)   # Indices of missing rows

            # Train GP on observed data
            
            gp = GaussianProcessRegressor(kernel=model_kernel, random_state=42)
            gp.fit(X_observed, y_observed)

            # Predict missing values
            y_missing_pred, _ = gp.predict(X_missing, return_std=True)

            # Fill in the missing values
            imputed_data[missing_indices, col] = y_missing_pred

    if is_dataframe:
        return pd.DataFrame(imputed_data, columns=data.columns, index=data.index)
    else:
        return imputed_data

    # gp_mae = mean_absolute_error(X_full, imputed_data)
    # gp_rmse = math.sqrt(mean_squared_error(X_full, imputed_data))

    #print(f"For {model_kernel} kernel, MAE: {gp_mae}, RMSE={gp_rmse}")

# Create a dataset with missing values
np.random.seed(42)
# mask = np.random.rand(*X.shape) > 0.2
# X_masked = X.copy()
# X_masked[mask] = np.nan

# Impute missing values
# imputed_data = gp_impute(X_masked)


# gp_mae = mean_absolute_error(X, imputed_data)
# gp_rmse = math.sqrt(mean_squared_error(X, imputed_data))

# print("MAE: ", gp_mae)
# print("RMSE: ", gp_rmse)

#rbf = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-1)
# gp_impute(X_masked, kernel=rbf)


# ------------------------- 
# changing kernel
# matern1_2 = Matern(nu=0.5)
#gp_impute(X_masked, matern1_2)

# matern3_2 = Matern(nu=1.5)
# gp_impute(X_masked, matern3_2)

# matern5_2 = Matern(nu=2.5)
# gp_impute(X_masked, matern5_2)

# rational_quadratic = RationalQuadratic()
# gp_impute(X_masked, rational_quadratic)


#--------------------------------
# Load full dataset
df_full = np.load("NSG_Application\\FullDataset.npz")
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

# X_rbf = gp_impute(X_full_masked, kernel=rbf)
# X_matern12 = gp_impute(X_full_masked, matern1_2)
# X_matern3_2 = gp_impute(X_full_masked, matern3_2)
# X_matern5_2 = gp_impute(X_full_masked, matern5_2)
# X_rq = gp_impute(X_full_masked, rational_quadratic)

# outputs
Y_full = df_full["array_2"]
Y_full_train = Y_full[:500]
Y_full_test = Y_full[500:]

#rbf
# X_rbf_train = X_rbf[:500, :]
# X_rbf_test = X_rbf[500:, :]

# gp_rbf = GaussianProcessClassifier(kernel = rbf)
# gp_rbf.fit(X_rbf_train, Y_full_train)
# gp_rbf_score = gp_rbf.score(X_rbf_test, Y_full_test)
# print("GP w/ RBF kernel score: ", gp_rbf_score)

# # matern 0.5
# X_matern12_train = X_matern12[:500, :]
# X_matern12_test = X_matern12[500:, :]

# gp_mat12 = GaussianProcessClassifier(kernel = matern1_2)
# gp_mat12.fit(X_matern12_train, Y_full_train)
# gp_mat12_score = gp_mat12.score(X_matern12_test, Y_full_test)
# print("GP w/ Matern 1/2 kernel score: ", gp_mat12_score)

# # matern 1.5
# X_matern32_train = X_matern3_2[:500, :]
# X_matern32_test = X_matern3_2[500:, :]

# gp_mat32 = GaussianProcessClassifier(kernel = matern3_2)
# gp_mat32.fit(X_matern32_train, Y_full_train)
# gp_mat32_score = gp_mat32.score(X_matern32_test, Y_full_test)
# print("GP w/ Matern 3/2 kernel score: ", gp_mat32_score)

# #matern 2.5
# X_matern52_train = X_matern5_2[:500, :]
# X_matern52_test = X_matern5_2[500:, :]

# gp_mat52 = GaussianProcessClassifier(kernel = matern5_2)
# gp_mat52.fit(X_matern52_train, Y_full_train)
# gp_mat52_score = gp_mat52.score(X_matern52_test, Y_full_test)
# print("GP w/ Matern 5/2 kernel score: ", gp_mat52_score)

# # rq
# X_rq_train = X_rq[:500, :]
# X_rq_test = X_rq[500:, :]

# gp_rq = GaussianProcessClassifier(kernel = rational_quadratic)
# gp_rq.fit(X_rq_train, Y_full_train)
# gp_rq_score = gp_rq.score(X_rq_test, Y_full_test)
# print("GP w/ RQ kernel score: ", gp_rq_score)


# DATASET ELEMENTS MISSING IN BLOCKS, SENSOR FAILURE ETC
# Data missing in blocks
X_block_missing = X_full  # 10 rows, 10 columns

# Create a mask for structured missingness (block missingness)
missing_fraction = 0.1  # Proportion of elements to remove per row
mask = np.ones_like(X_block_missing, dtype=bool)  # Start with all True (no missing values)

for col in range(X_block_missing.shape[1]):  # For each column
    # Select a random range of rows to mask
    num_missing = int(missing_fraction * X_block_missing.shape[0])  # Number of elements to mask
    start_idx = np.random.randint(0, X_block_missing.shape[0] - num_missing + 1)
    mask[start_idx:start_idx + num_missing, col] = False  # Mask a vertical section

# Apply the mask
X_block_missing_masked = X_block_missing.copy()
X_block_missing_masked[~mask] = np.nan

rbf = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-1)
# gp_impute(X_masked, kernel=rbf)

# ------------------------- 
# changing kernel
matern1_2 = Matern(nu=0.5)
#gp_impute(X_masked, matern1_2)

matern3_2 = Matern(nu=1.5)
# gp_impute(X_masked, matern3_2)

matern5_2 = Matern(nu=2.5)
# gp_impute(X_masked, matern5_2)

rational_quadratic = RationalQuadratic()
# gp_impute(X_masked, rational_quadratic)

X_rbf = gp_impute(X_block_missing_masked, kernel=rbf)
X_matern12 = gp_impute(X_block_missing_masked, matern1_2)
X_matern3_2 = gp_impute(X_block_missing_masked, matern3_2)
X_matern5_2 = gp_impute(X_block_missing_masked, matern5_2)
X_rq = gp_impute(X_block_missing_masked, rational_quadratic)

# outputs
Y_full = df_full["array_2"]
Y_full_train = Y_full[:500]
Y_full_test = Y_full[500:]

#rbf
X_rbf_train = X_rbf[:500, :]
X_rbf_test = X_rbf[500:, :]

# gp_rbf = GaussianProcessClassifier(kernel = rbf)
# gp_rbf.fit(X_rbf_train, Y_full_train)
# gp_rbf_score = gp_rbf.score(X_rbf_test, Y_full_test)
# print("GP w/ RBF kernel score: ", gp_rbf_score)

# matern 0.5
X_matern12_train = X_matern12[:500, :]
X_matern12_test = X_matern12[500:, :]

# gp_mat12 = GaussianProcessClassifier(kernel = matern1_2)
# gp_mat12.fit(X_matern12_train, Y_full_train)
# gp_mat12_score = gp_mat12.score(X_matern12_test, Y_full_test)
# print("GP w/ Matern 1/2 kernel score: ", gp_mat12_score)

# matern 1.5
X_matern32_train = X_matern3_2[:500, :]
X_matern32_test = X_matern3_2[500:, :]

# gp_mat32 = GaussianProcessClassifier(kernel = matern3_2)
# gp_mat32.fit(X_matern32_train, Y_full_train)
# gp_mat32_score = gp_mat32.score(X_matern32_test, Y_full_test)
# print("GP w/ Matern 3/2 kernel score: ", gp_mat32_score)

#matern 2.5
X_matern52_train = X_matern5_2[:500, :]
X_matern52_test = X_matern5_2[500:, :]

# gp_mat52 = GaussianProcessClassifier(kernel = matern5_2)
# gp_mat52.fit(X_matern52_train, Y_full_train)
# gp_mat52_score = gp_mat52.score(X_matern52_test, Y_full_test)
# print("GP w/ Matern 5/2 kernel score: ", gp_mat52_score)

# rq
X_rq_train = X_rq[:500, :]
X_rq_test = X_rq[500:, :]

# gp_rq = GaussianProcessClassifier(kernel = rational_quadratic)
# gp_rq.fit(X_rq_train, Y_full_train)
# gp_rq_score = gp_rq.score(X_rq_test, Y_full_test)
# print("GP w/ RQ kernel score: ", gp_rq_score)


# GP (Classifier Kernel) (Imputation Kernel) 
gp_rbf_rbf = GaussianProcessClassifier(kernel=rbf)
gp_rbf_mat12= GaussianProcessClassifier(kernel=matern1_2)
gp_rbf_mat32= GaussianProcessClassifier(kernel=matern3_2)
gp_rbf_mat52= GaussianProcessClassifier(kernel=matern5_2)
gp_rbf_rq= GaussianProcessClassifier(kernel=rational_quadratic)
print("Created gp_rbf_(kernel)s")

gp_mat12_rbf = GaussianProcessClassifier(kernel=rbf)
gp_mat12_mat12= GaussianProcessClassifier(kernel=matern1_2)
gp_mat12_mat32= GaussianProcessClassifier(kernel=matern3_2)
gp_mat12_mat52= GaussianProcessClassifier(kernel=matern5_2)
gp_mat12_rq= GaussianProcessClassifier(kernel=rational_quadratic)
print("Created gp_mat12_(kernel)s")

gp_mat32_rbf= GaussianProcessClassifier(kernel=rbf)
gp_mat32_mat12= GaussianProcessClassifier(kernel=matern1_2)
gp_mat32_mat32= GaussianProcessClassifier(kernel=matern3_2)
gp_mat32_mat52= GaussianProcessClassifier(kernel=matern5_2)
gp_mat32_rq= GaussianProcessClassifier(kernel=rational_quadratic)
print("Created gp_mat32_(kernel)s")

gp_mat52_rbf= GaussianProcessClassifier(kernel=rbf)
gp_mat52_mat12= GaussianProcessClassifier(kernel=matern1_2)
gp_mat52_mat32= GaussianProcessClassifier(kernel=matern3_2)
gp_mat52_mat52= GaussianProcessClassifier(kernel=matern5_2)
gp_mat52_rq= GaussianProcessClassifier(kernel=rational_quadratic)
print("Created gp_mat52_(kernel)s")

gp_rq_rbf= GaussianProcessClassifier(kernel=rbf)
gp_rq_mat12= GaussianProcessClassifier(kernel=matern1_2)
gp_rq_mat32= GaussianProcessClassifier(kernel=matern3_2)
gp_rq_mat52= GaussianProcessClassifier(kernel=matern5_2)
gp_rq_rq= GaussianProcessClassifier(kernel=rational_quadratic)
print("Created gp_rq_(kernel)s")

# train gpcs
gp_rbf_rbf.fit(X_rbf_train, Y_full_train)         # gp using rbf kernel to classify data that was imputed using rbf kernel
gp_rbf_mat12.fit(X_matern12_train, Y_full_train)  # gp using rbf kernel to classify data that was imputed using matern1/2 kernel
gp_rbf_mat32.fit(X_matern32_train, Y_full_train)
gp_rbf_mat52.fit(X_matern52_train, Y_full_train)
gp_rbf_rq.fit(X_rq_train, Y_full_train)

gp_mat12_rbf.fit(X_rbf_train, Y_full_train)
gp_mat12_mat12.fit(X_matern12_train, Y_full_train)
gp_mat12_mat32.fit(X_matern32_train, Y_full_train)
gp_mat12_mat52.fit(X_matern52_train, Y_full_train)
gp_mat12_rq.fit(X_rq_train, Y_full_train)

gp_mat32_rbf.fit(X_rbf_train, Y_full_train)
gp_mat32_mat12.fit(X_matern12_train, Y_full_train)
gp_mat32_mat32.fit(X_matern32_train, Y_full_train)
gp_mat32_mat52.fit(X_matern52_train, Y_full_train)
gp_mat32_rq.fit(X_rq_train, Y_full_train)

gp_mat52_rbf.fit(X_rbf_train, Y_full_train)
gp_mat52_mat12.fit(X_matern12_train, Y_full_train)
gp_mat52_mat32.fit(X_matern32_train, Y_full_train)
gp_mat52_mat52.fit(X_matern52_train, Y_full_train)
gp_mat52_rq.fit(X_rq_train, Y_full_train)

gp_rq_rbf.fit(X_rbf_train, Y_full_train)
gp_rq_mat12.fit(X_matern12_train, Y_full_train)
gp_rq_mat32.fit(X_matern32_train, Y_full_train)
gp_rq_mat52.fit(X_matern52_train, Y_full_train)
gp_rq_rq.fit(X_rq_train, Y_full_train)


# score gps
gp_rbf_rbf_score = gp_rbf_rbf.score(X_rbf_train,Y_full_train)
gp_rbf_mat12_score = gp_rbf_mat12.score(X_matern12_train, Y_full_train)
gp_rbf_mat32_score = gp_rbf_mat32.score(X_matern32_train, Y_full_train)
gp_rbf_mat52_score = gp_rbf_mat52.score(X_matern52_train, Y_full_train)
gp_rbf_rq_score = gp_rbf_rq.score(X_rq_train, Y_full_train)

gp_mat12_rbf_score = gp_mat12_rbf.score(X_rbf_train, Y_full_train)
gp_mat12_mat12_score = gp_mat12_mat12.score(X_matern12_train, Y_full_train)
gp_mat12_mat32_score = gp_mat12_mat32.score(X_matern32_train, Y_full_train)
gp_mat12_mat52_score = gp_mat12_mat52.score(X_matern52_train, Y_full_train)
gp_mat12_rq_score = gp_mat12_rq.score(X_rq_train, Y_full_train)

gp_mat32_rbf_score = gp_mat32_rbf.score(X_rbf_train, Y_full_train)
gp_mat32_mat12_score = gp_mat32_mat12.score(X_matern12_train, Y_full_train)
gp_mat32_mat32_score = gp_mat32_mat32.score(X_matern32_train, Y_full_train)
gp_mat32_mat52_score = gp_mat32_mat52.score(X_matern52_train, Y_full_train)
gp_mat32_rq_score = gp_mat32_rq.score(X_rq_train, Y_full_train)

gp_mat52_rbf_score = gp_mat32_rbf.score(X_rbf_train, Y_full_train)
gp_mat52_mat12_score = gp_mat32_mat12.score(X_matern12_train, Y_full_train)
gp_mat52_mat32_score = gp_mat32_mat32.score(X_matern32_train, Y_full_train)
gp_mat52_mat52_score = gp_mat32_mat52.score(X_matern52_train, Y_full_train)
gp_mat52_rq_score = gp_mat32_rq.score(X_rq_train, Y_full_train)

gp_rq_rbf_score = gp_rq_rbf.score(X_rbf_train, Y_full_train)
gp_rq_mat12_score = gp_rq_mat12.score(X_matern12_train, Y_full_train)
gp_rq_mat32_score = gp_rq_mat32.score(X_matern32_train, Y_full_train)
gp_rq_mat52_score = gp_rq_mat52.score(X_matern52_train, Y_full_train)
gp_rq_rq_score = gp_rq_rq.score(X_rq_train, Y_full_train)

print("done")