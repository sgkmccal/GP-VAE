import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

df = pd.read_csv("NSG_Application\\temp_passfail_data.csv")
df = df.iloc[1:, 3:58]
df_shape = df.shape

df_subset = df.iloc[:60, :]
df_subset_shape = df_subset.shape

np.random.seed(1234)
missing_rate = 0.2
missing_vals_mask_full = np.random.rand(*df_shape) > 0.4
missing_vals_mask = np.random.rand(*df_subset_shape) > 0.4
print(missing_vals_mask)

df_full_masked = df * missing_vals_mask_full
df_subset_masked = df_subset * missing_vals_mask
print(df_subset_masked)

# Visualize the MNAR mask
def plot_boolean_mask(mask, title="MNAR"):
    rows, cols = mask.shape
    fig, ax = plt.subplots(figsize=(cols / 10, rows / 10))
    for row in range(rows):
        for col in range(cols):
            color = 'green' if mask[row, col] else 'red'
            rect = plt.Rectangle((col, rows - row - 1), 1, 1, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(title)
    plt.show()

# Pad start and end of df_full_masked so 0s on edges are handled
df_full_masked.ffill(axis=1, inplace=True)
df_full_masked.bfill(axis=1, inplace=True)

# Perform linear interpolation
nsg_df_reconstructed_linear_interp = df_full_masked.interpolate(method='linear', axis=1)

# Handle any remaining NaNs after interpolation
nsg_df_reconstructed_linear_interp = nsg_df_reconstructed_linear_interp.bfill(axis=1)
nsg_df_reconstructed_linear_interp = nsg_df_reconstructed_linear_interp.ffill(axis=1)

print("Linear interpolation:", nsg_df_reconstructed_linear_interp)

# Masking missing values (e.g., where values are 0.0)
mask = df != 0.0  # Create mask for non-zero values

# Calculate the average of each column, ignoring zeros
column_averages = df.where(mask).mean(axis=0)

# Now calculate MAE and RMSE for each column
mae_list = []
rmse_list = []

# Iterate over each column to calculate MAE and RMSE
for column in df.columns:
    # True values are the original data column
    true_values = df[column]
    
    # Predicted values are the column averages (scalar)
    predicted_values = column_averages[column]
    
    # Mask out zeros for comparison
    mask_column = mask[column]
    
    # Select non-zero true values
    true_values_filtered = true_values[mask_column]
    
    # For non-zero true values, predicted values are the same (column average)
    predicted_values_filtered = np.full_like(true_values_filtered, predicted_values)
    
    # Calculate MAE and RMSE for this column
    mae = mean_absolute_error(true_values_filtered, predicted_values_filtered)
    rmse = math.sqrt(mean_squared_error(true_values_filtered, predicted_values_filtered))
    
    # Append to lists
    mae_list.append(mae)
    rmse_list.append(rmse)

# Average the MAE and RMSE across all columns
average_mae = np.mean(mae_list)
average_rmse = np.mean(rmse_list)

# Calculate MAE and RMSE as percentages based on the range of values (assuming range is 500-650)
range_of_values = 650 - 500  # Adjust this range to match your data's range
average_mae_percentage = (average_mae / range_of_values) * 100
average_rmse_percentage = (average_rmse / range_of_values) * 100

# Print results
print(f"Average MAE (percentage): {average_mae_percentage:.2f}%")
print(f"Average RMSE (percentage): {average_rmse_percentage:.2f}%")

# Compare the original data and the reconstructed (interpolated) data
original_reconstructed_mae_linear = mean_absolute_error(df, nsg_df_reconstructed_linear_interp)
original_reconstructed_rmse_linear = math.sqrt(mean_squared_error(df, nsg_df_reconstructed_linear_interp))
print(f"Original vs Reconstructed MAE: {original_reconstructed_mae_linear}")
print(f"Original vs Reconstructed RMSE: {original_reconstructed_rmse_linear}")
