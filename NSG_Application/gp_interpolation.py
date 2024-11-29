import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from sklearn.preprocessing import StandardScaler

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

    # if is_dataframe:
    #     return pd.DataFrame(imputed_data, columns=data.columns, index=data.index)
    # else:
    #     return imputed_data

    gp_mae = mean_absolute_error(X, imputed_data)
    gp_rmse = math.sqrt(mean_squared_error(X, imputed_data))

    print(f"For {model_kernel} kernel, MAE: {gp_mae}, RMSE={gp_rmse}")

# Create a dataset with missing values
np.random.seed(42)
mask = np.random.rand(*X.shape) > 0.2
X_masked = X.copy()
X_masked[mask] = np.nan

# Impute missing values
# imputed_data = gp_impute(X_masked)


# gp_mae = mean_absolute_error(X, imputed_data)
# gp_rmse = math.sqrt(mean_squared_error(X, imputed_data))

# print("MAE: ", gp_mae)
# print("RMSE: ", gp_rmse)

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
gp_impute(X_masked, rational_quadratic)

