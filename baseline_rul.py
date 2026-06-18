import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# Column Names
# ==========================================

columns = [
    "engine_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3"
]

columns += [f"sensor_{i}" for i in range(1, 22)]

# ==========================================
# Load Training Data
# ==========================================

train_df = pd.read_csv(
    "data/train_FD001.txt",
    sep=r"\s+",
    header=None
)

train_df = train_df.iloc[:, :26]
train_df.columns = columns

# ==========================================
# Compute RUL Labels
# ==========================================

max_cycles = train_df.groupby("engine_id")["cycle"].max()

train_df["RUL"] = (
    train_df["engine_id"].map(max_cycles)
    - train_df["cycle"]
)

# Clip RUL at 125
train_df["RUL"] = train_df["RUL"].clip(upper=125)

# ==========================================
# Feature Selection
# ==========================================

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

constant_sensors = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19"
]

feature_cols = [
    col
    for col in sensor_cols
    if col not in constant_sensors
]

print("Removed constant sensors:")
print(constant_sensors)

X_train = train_df[feature_cols]
y_train = train_df["RUL"]

# ==========================================
# Train Random Forest
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# Load Test Data
# ==========================================

test_df = pd.read_csv(
    "data/test_FD001.txt",
    sep=r"\s+",
    header=None
)

test_df = test_df.iloc[:, :26]
test_df.columns = columns

# ==========================================
# Last Cycle Per Engine
# ==========================================

test_last = (
    test_df
    .sort_values(["engine_id", "cycle"])
    .groupby("engine_id")
    .tail(1)
)

X_test = test_last[feature_cols]

# ==========================================
# Load Ground Truth
# ==========================================

y_test = pd.read_csv(
    "data/RUL_FD001.txt",
    sep=r"\s+",
    header=None
)

y_test = y_test.iloc[:, 0].values

# ==========================================
# Predict
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# Evaluate
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

print("\n========================")
print(f"Test set RMSE: {rmse:.2f}")
print(f"Test set MAE : {mae:.2f}")
print("========================")