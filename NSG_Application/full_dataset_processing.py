# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 08:12:19 2024

@author: sgkmccal
"""

import numpy as np
import pandas as pd

df = pd.read_csv(
    "NSG_Application\\combined_data_gsmod08112022_FULLDATASET.csv")

columns = """
ScanDateTimeGlasses	Gap	ScanDateTimeFurnace	LineSpeed	

ZN01_TOP_Temp	ZN01_BOTTOM_TEMP\
ZN01_TOP_OUTPUT	ZN01_BOTTOM_OUTPUT	ZN02_TOP_Temp	ZN02_BOTTOM_TEMP	ZN02_TOP_OUTPUT	ZN02_BOTTOM_OUTPUT	
ZN03_TOP_Temp	ZN03_BOTTOM_TEMP	ZN03_TOP_OUTPUT	ZN03_BOTTOM_OUTPUT	ZN04_TOP_Temp	ZN04_BOTTOM_TEMP	
ZN04_TOP_OUTPUT	ZN04_BOTTOM_OUTPUT	ZN05_TOP_Temp	ZN05_BOTTOM_TEMP	ZN05_TOP_OUTPUT	ZN05_BOTTOM_OUTPUT	
ZN06_TOP_Temp	ZN06_BOTTOM_TEMP	ZN06_TOP_OUTPUT	ZN06_BOTTOM_OUTPUT	ZN07_TOP_Temp	ZN07_BOTTOM_TEMP	
ZN07_TOP_OUTPUT	ZN07_BOTTOM_OUTPUT	ZN08_TOP_Temp	ZN08_BOTTOM_TEMP	ZN08_TOP_OUTPUT	ZN08_BOTTOM_OUTPUT	
ZN09_E01_TEMP	ZN09_E02_TEMP	ZN09_E03_TEMP	ZN09_E04_TEMP	ZN09_E05_TEMP	ZN09_E06_TEMP	
ZN09_E07_TEMP	ZN09_E08_TEMP	ZN09_E09_TEMP	ZN09_E10_TEMP	ZN09_E11_TEMP	ZN09_E12_TEMP	
ZN09_BOTTOM_TEMP	ZN09_BOTTOM_OUTPUT	ZN10_E01_TEMP	ZN10_E02_TEMP	ZN10_E03_TEMP	ZN10_E04_TEMP	
ZN10_E05_TEMP	ZN10_E06_TEMP	ZN10_E07_TEMP	ZN10_E08_TEMP	ZN10_E09_TEMP	ZN10_E10_TEMP	ZN10_E11_TEMP
ZN10_E12_TEMP	ZN10_BOTTOM_TEMP	ZN10_BOTTOM_OUTPUT	ZN11_E01_TEMP	ZN11_E02_TEMP	ZN11_E03_TEMP	
ZN11_E04_TEMP	ZN11_E05_TEMP	ZN11_E06_TEMP	ZN11_E07_TEMP	ZN11_E08_TEMP	ZN11_E09_TEMP	ZN11_E10_TEMP
ZN11_E11_TEMP	ZN11_E12_TEMP	ZN11_BOTTOM_TEMP	ZN11_BOTTOM_OUTPUT	

ScanDateTimePress	

PressPos_IG	PressPos_OG	

ScanDateTimeAirFloat	

Temp_Stage1_Element1	Temp_Stage1_Element2	Temp_Stage1_Element3	
Temp_Stage1_Element4	Temp_Stage1_Output	Temp_Stage2_Element1	Temp_Stage2_Element2	
Temp_Stage2_Element3	Temp_Stage2_Element4	Temp_Stage2_Output	AirPressure_In	AirPressure_Out	

ScanDateTimeResistance	

GlasTemperature	Resistance1	Resistance2	

ScanDateTimeRaytek	

Max1Press	Min1Press	
Max2Press	Min2Press	Max3Press	Min3Press	Max4Press	Min4Press	Max5Press	Min5Press	Max6Press	
Min6Press	Max7Press	Min7Press	Max8Press	Min8Press	Max9Press	Min9Press	MaxRefPress	MinRefPress	
Max1Shuttle	Min1Shuttle	Max2Shuttle	Min2Shuttle	Max3Shuttle	Min3Shuttle	Max4Shuttle	Min4Shuttle	Max5Shuttle	
Min5Shuttle	Max6Shuttle	Min6Shuttle	Max7Shuttle	Min7Shuttle	Max8Shuttle	Min8Shuttle	
Max9Shuttle	Min9Shuttle	MaxRefShuttle	MinRefShuttle	GlassID1 2	TimeID	HUD Value	Classification
"""

# Split columns and add speech marks
columns_list = columns.split("\t")
formatted_columns = [f'"{col}"' for col in columns_list]

# Join into a comma-separated string
formatted_output = ", ".join(formatted_columns)

# Print or save the result
print(formatted_output)

# more input features
"""
PressPos_IG	PressPos_OG	 Temp_Stage1_Element1	Temp_Stage1_Element2	Temp_Stage1_Element3	
Temp_Stage1_Element4	Temp_Stage1_Output	Temp_Stage2_Element1	Temp_Stage2_Element2	
Temp_Stage2_Element3	Temp_Stage2_Element4	Temp_Stage2_Output	AirPressure_In	AirPressure_Out	
GlasTemperature	Resistance1	Resistance2	
Max1Press	Min1Press	
Max2Press	Min2Press	Max3Press	Min3Press	Max4Press	Min4Press	Max5Press	Min5Press	Max6Press	
Min6Press	Max7Press	Min7Press	Max8Press	Min8Press	Max9Press	Min9Press	MaxRefPress	MinRefPress	
Max1Shuttle	Min1Shuttle	Max2Shuttle	Min2Shuttle	Max3Shuttle	Min3Shuttle	Max4Shuttle	Min4Shuttle	Max5Shuttle	
Min5Shuttle	Max6Shuttle	Min6Shuttle	Max7Shuttle	Min7Shuttle	Max8Shuttle	Min8Shuttle	
Max9Shuttle	Min9Shuttle	MaxRefShuttle	MinRefShuttle
"""

# strip last row (average of each column to check for constants)
df = df.iloc[:-1, :]

# Converting times to unix time
df["ScanDateTimeGlasses"] = pd.to_datetime(df["ScanDateTimeGlasses"])
df["ScanDateTimeFurnace"] = pd.to_datetime(df["ScanDateTimeFurnace"])
df["ScanDateTimePress"] = pd.to_datetime(df["ScanDateTimePress"])
df["ScanDateTimeAirFloat"] = pd.to_datetime(df["ScanDateTimeAirFloat"])
df["ScanDateTimeResistance"] = pd.to_datetime(df["ScanDateTimeResistance"])
df["ScanDateTimeRaytek"] = pd.to_datetime(df["ScanDateTimeRaytek"])

# Convert to integer (seconds since epoch)
df["ScanDateTimeGlasses"] = df["ScanDateTimeGlasses"].view("int64") // 10**9
df["ScanDateTimeFurnace"] = df["ScanDateTimeFurnace"].view("int64") // 10**9
df["ScanDateTimePress"] = df["ScanDateTimePress"].view("int64") // 10**9
df["ScanDateTimeAirFloat"] = df["ScanDateTimeAirFloat"].view("int64") // 10**9
df["ScanDateTimeResistance"] = df["ScanDateTimeResistance"].view(
    "int64") // 10**9
df["ScanDateTimeRaytek"] = df["ScanDateTimeRaytek"].view("int64") // 10**9

# scale back to zero
# df["ScanDateTimeGlasses"] = df["ScanDateTimeGlasses"] - df["ScanDateTimeGlasses"].iloc[0]
# df["ScanDateTimeFurnace"] = df["ScanDateTimeFurnace"] - df["ScanDateTimeFurnace"].iloc[0]
# df["ScanDateTimePress"] = df["ScanDateTimePress"] - df["ScanDateTimePress"].iloc[0]
# df["ScanDateTimeAirFloat"] = df["ScanDateTimeAirFloat"] - df["ScanDateTimeAirFloat"].iloc[0]
# df["ScanDateTimeResistance"] = df["ScanDateTimeResistance"] - df["ScanDateTimeResistance"].iloc[0]
# df["ScanDateTimeRaytek"] = df["ScanDateTimeRaytek"] - df["ScanDateTimeRaytek"].iloc[0]

df["Glasses_TimeDifference"] = df["ScanDateTimeGlasses"].diff()
df["Furnace_TimeDifference"] = df["ScanDateTimeFurnace"].diff()
df["Press_TimeDifference"] = df["ScanDateTimePress"].diff()
df["Resistance_TimeDifference"] = df["ScanDateTimeResistance"].diff()
df["Raytek_TimeDifference"] = df["ScanDateTimeRaytek"].diff()

# no longer needed now using .diff()
df.drop(["ScanDateTimeGlasses", "ScanDateTimeFurnace", "ScanDateTimePress", "ScanDateTimeAirFloat",
         "ScanDateTimeResistance", "ScanDateTimeRaytek"], axis=1, inplace=True)

# first elem contains NaNs due to diff() so dropping
df = df.iloc[1:, :]

X = df[["ZN01_TOP_Temp", "ZN01_BOTTOM_TEMP", "ZN01_TOP_OUTPUT", "ZN01_BOTTOM_OUTPUT", "ZN02_TOP_Temp", "ZN02_BOTTOM_TEMP",
       "ZN02_TOP_OUTPUT", "ZN02_BOTTOM_OUTPUT",
        "ZN03_TOP_Temp", "ZN03_BOTTOM_TEMP", "ZN03_TOP_OUTPUT", "ZN03_BOTTOM_OUTPUT", "ZN04_TOP_Temp", "ZN04_BOTTOM_TEMP",
        "ZN04_TOP_OUTPUT", "ZN04_BOTTOM_OUTPUT", "ZN05_TOP_Temp", "ZN05_BOTTOM_TEMP", "ZN05_TOP_OUTPUT", "ZN05_BOTTOM_OUTPUT",
        "ZN06_TOP_Temp", "ZN06_BOTTOM_TEMP", "ZN06_TOP_OUTPUT", "ZN06_BOTTOM_OUTPUT", "ZN07_TOP_Temp", "ZN07_BOTTOM_TEMP",
        "ZN07_TOP_OUTPUT", "ZN07_BOTTOM_OUTPUT", "ZN08_TOP_Temp", "ZN08_BOTTOM_TEMP", "ZN08_TOP_OUTPUT", "ZN08_BOTTOM_OUTPUT",
        "ZN09_E01_TEMP", "ZN09_E02_TEMP", "ZN09_E03_TEMP", "ZN09_E04_TEMP", "ZN09_E05_TEMP", "ZN09_E06_TEMP",
        "ZN09_E07_TEMP", "ZN09_E08_TEMP", "ZN09_E09_TEMP", "ZN09_E10_TEMP", "ZN09_E11_TEMP", "ZN09_E12_TEMP",
        "ZN09_BOTTOM_TEMP", "ZN09_BOTTOM_OUTPUT", "ZN10_E01_TEMP", "ZN10_E02_TEMP", "ZN10_E03_TEMP", "ZN10_E04_TEMP",
        "ZN10_E05_TEMP", "ZN10_E06_TEMP", "ZN10_E07_TEMP", "ZN10_E08_TEMP", "ZN10_E09_TEMP", "ZN10_E10_TEMP", "ZN10_E11_TEMP",
        "ZN10_E12_TEMP", "ZN10_BOTTOM_TEMP", "ZN10_BOTTOM_OUTPUT", "ZN11_E01_TEMP", "ZN11_E02_TEMP", "ZN11_E03_TEMP",
        "ZN11_E04_TEMP", "ZN11_E05_TEMP", "ZN11_E06_TEMP", "ZN11_E07_TEMP", "ZN11_E08_TEMP", "ZN11_E09_TEMP", "ZN11_E10_TEMP",
        "ZN11_E11_TEMP", "ZN11_E12_TEMP", "ZN11_BOTTOM_TEMP", "ZN11_BOTTOM_OUTPUT",
        "Max1Press", "Min1Press", "Max2Press", "Min2Press", "Max3Press", "Min3Press", "Max4Press", "Min4Press", "Max5Press",
        "Min5Press", "Max6Press", "Min6Press", "Max7Press", "Min7Press", "Max8Press", "Min8Press", "Max9Press", "Min9Press",
        "MaxRefPress", "MinRefPress", "Max1Shuttle", "Min1Shuttle", "Max2Shuttle", "Min2Shuttle", "Max3Shuttle", "Min3Shuttle",
        "Max4Shuttle", "Min4Shuttle", "Max5Shuttle", "Min5Shuttle", "Max6Shuttle", "Min6Shuttle", "Max7Shuttle", "Min7Shuttle",
        "Max8Shuttle", "Min8Shuttle", "Max9Shuttle", "Min9Shuttle", "MaxRefShuttle", "MinRefShuttle"]]

Y = df[["Classification"]]

X_arr = np.asarray(X)
Y_arr = np.asarray(Y)

np.savez("FullDataset", array1=X_arr, array_2=Y_arr)


# ----------------------
# Data missing in blocks
X_block_missing = X_arr  # 10 rows, 10 columns

# Create a mask for structured missingness (block missingness)
missing_fraction = 0.1  # Proportion of elements to remove per row
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
