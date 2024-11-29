# GP & LR Classifiers applied to original data

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
 
# --- Path to dataset file ---
path = 'C:\\Users\\Kieron\\Documents\\GitHub\\GP-VAE\\GP-VAE NSG Data\\Data\\original_data.csv'
 
temp_col_names = [
    "ZN01_TOP_Temp", "ZN01_BOTTOM_TEMP", "ZN02_TOP_Temp", "ZN02_BOTTOM_TEMP",
    "ZN03_TOP_Temp", "ZN03_BOTTOM_TEMP", "ZN04_TOP_Temp", "ZN04_BOTTOM_TEMP",
    "ZN05_TOP_Temp", "ZN05_BOTTOM_TEMP", "ZN06_TOP_Temp", "ZN06_BOTTOM_TEMP",
    "ZN07_TOP_Temp", "ZN07_BOTTOM_TEMP", "ZN08_TOP_Temp", "ZN08_BOTTOM_TEMP",
    "ZN09_E01_TEMP", "ZN09_E02_TEMP", "ZN09_E03_TEMP", "ZN09_E04_TEMP",
    "ZN09_E05_TEMP", "ZN09_E06_TEMP", "ZN09_E07_TEMP", "ZN09_E08_TEMP",
    "ZN09_E09_TEMP", "ZN09_E10_TEMP", "ZN09_E11_TEMP", "ZN09_E12_TEMP",
    "ZN09_BOTTOM_TEMP", "ZN10_E01_TEMP", "ZN10_E02_TEMP", "ZN10_E03_TEMP",
    "ZN10_E04_TEMP", "ZN10_E05_TEMP", "ZN10_E06_TEMP", "ZN10_E07_TEMP",
    "ZN10_E08_TEMP", "ZN10_E09_TEMP", "ZN10_E10_TEMP", "ZN10_E11_TEMP",
    "ZN10_E12_TEMP", "ZN10_BOTTOM_TEMP", "ZN11_E01_TEMP", "ZN11_E02_TEMP",
    "ZN11_E03_TEMP", "ZN11_E04_TEMP", "ZN11_E05_TEMP", "ZN11_E06_TEMP",
    "ZN11_E07_TEMP", "ZN11_E08_TEMP", "ZN11_E09_TEMP", "ZN11_E10_TEMP",
    "ZN11_E11_TEMP", "ZN11_E12_TEMP", "ZN11_BOTTOM_TEMP"
]
 
D = pd.read_csv(path, index_col=0)
X = np.asarray(D.loc[:, temp_col_names].iloc[:771, :])
Y = np.asarray(D.loc[:, "Class"].iloc[:771])
 
train_pc_dataset = 0.8  # Percentage of dataset to be used for training set
n_training_samples = int(X.shape[0] * train_pc_dataset)
n_testing_samples = X.shape[0] - n_training_samples
n_features = X.shape[1]
 
 
scaler = StandardScaler()
X = scaler.fit_transform(X)
x_train = X[:n_training_samples, :]
x_test = X[n_training_samples:, :]
y_train = Y[:n_training_samples]
y_test = Y[n_training_samples:]
 
gpc = GPC()
gpc.fit(x_train, y_train)
gpc_score = gpc.score(x_test, y_test)
 
print(f"For {n_training_samples} training samples, the GP model predicts the output \
of {n_testing_samples} with an accuracy of {gpc_score*100:.2f}%")