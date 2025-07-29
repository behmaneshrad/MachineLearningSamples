# ===========================================================
# Urban Traffic Volume Prediction – COMPLETE Python Script
# ===========================================================
# This single file can be executed in any Python 3.8+ environment
# (Jupyter, VS Code, PyCharm, etc.) after installing the required libs:
#
#   pip install pandas numpy matplotlib seaborn scikit-learn joblib
#
# Assumptions
# -----------
# 1. A folder named `dataset` exists in the same directory as this script.
# 2. Inside `dataset`, the file `x.csv` contains the 100-row sample posted
#    in the prompt.
#
# What the script does
# --------------------
# • Loads the raw data
# • Performs all preprocessing described in the prompt
# • Removes outliers via IQR
# • Builds a Ridge-regularised linear regression pipeline
# • Prints parameters & metrics at every stage
# • Generates informative plots for visual inspection
# • Saves the trained pipeline to disk
# ===========================================================

# ------------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------------
import os, warnings, json, datetime, math
warnings.filterwarnings("ignore")               # Cleaner output

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV

import joblib  # For model persistence

# ------------------------------------------------------------------
# 1. Global style tweaks for prettier plots
# ------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 4)

# ------------------------------------------------------------------
# 2. Load the dataset
# ------------------------------------------------------------------
DATA_PATH = os.path.join("dataset", "Metro_Interstate_Traffic_Volume.csv")
df = pd.read_csv(DATA_PATH)

print("\n" + "="*60)
print("Step 1 – Raw Data Loaded")
print("="*60)
print(f"Shape: {df.shape}")
print("\nHead:")
display(df.head())

# ------------------------------------------------------------------
# 3. Basic Pre-processing
# ------------------------------------------------------------------
# 3-a. Convert date_time to pandas datetime
df["date_time"] = pd.to_datetime(df["date_time"], format="%m/%d/%Y %H:%M")

# 3-b. Extract temporal features
df["hour"]    = df["date_time"].dt.hour
df["weekday"] = df["date_time"].dt.day_name()
df["month"]   = df["date_time"].dt.month

# 3-c. Temperature conversion: Kelvin → Celsius
df["temp_c"] = df["temp"] - 273.15

print("\n" + "="*60)
print("Step 2 – Basic Pre-processing Complete")
print("="*60)
print("Temp range in °C: %.1f – %.1f" % (df.temp_c.min(), df.temp_c.max()))
print("\nExtracted datetime features:")
display(df[["date_time", "hour", "weekday", "month"]].head())

# ------------------------------------------------------------------
# 4. Outlier Removal (IQR on target variable)
# ------------------------------------------------------------------
q25, q75 = np.percentile(df.traffic_volume, [25, 75])
iqr = q75 - q25
lower_bound = q25 - 1.5 * iqr
upper_bound = q75 + 1.5 * iqr

mask = df.traffic_volume.between(lower_bound, upper_bound)
df_clean = df[mask].reset_index(drop=True)

print("\n" + "="*60)
print("Step 3 – Outlier Removal")
print("="*60)
print(f"Rows before: {len(df)}")
print(f"Rows after : {len(df_clean)}")
print(f"IQR bounds : [{lower_bound:.1f}, {upper_bound:.1f}]")

# Quick visual verification
fig, ax = plt.subplots(1, 2, figsize=(11, 3))
sns.boxplot(y=df.traffic_volume, ax=ax[0], color="skyblue")
ax[0].set_title("Before outlier removal")

sns.boxplot(y=df_clean.traffic_volume, ax=ax[1], color="salmon")
ax[1].set_title("After outlier removal")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 5. Train / Test Split
# ------------------------------------------------------------------
y = df_clean["traffic_volume"]
X = df_clean.drop(columns=["traffic_volume", "temp", "date_time"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("\n" + "="*60)
print("Step 4 – Train/Test Split")
print("="*60)
print(f"Train samples: {len(X_train)}")
print(f"Test  samples: {len(X_test)}")

# ------------------------------------------------------------------
# 6. Build preprocessing & modelling pipeline
# ------------------------------------------------------------------
# Column lists
cat_features = ["holiday", "weather_main", "weather_description", "weekday"]
num_features = ["temp_c", "rain_1h", "snow_1h", "clouds_all", "hour", "month"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features)
    ]
)

# Ridge regression with built-in cross-validation for alpha
alphas = np.logspace(-3, 3, 20)  # 0.001 … 1000
ridge = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_absolute_error")

# Full pipeline
model = Pipeline(
    steps=[("prep", preprocessor),
           ("reg", ridge)]
)

print("\n" + "="*60)
print("Step 5 – Pipeline Constructed")
print("="*60)
print("Categorical features :", cat_features)
print("Numerical   features :", num_features)
print("Candidate alphas     :", alphas.round(3))

# ------------------------------------------------------------------
# 7. Fit Model
# ------------------------------------------------------------------
model.fit(X_train, y_train)

print("\n" + "="*60)
print("Step 6 – Model Training Complete")
print("="*60)
print("Chosen alpha via CV : %.3f" % model.named_steps["reg"].alpha_)

# ------------------------------------------------------------------
# 8. Evaluation
# ------------------------------------------------------------------
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("Step 7 – Evaluation on Test Set")
print("="*60)
eval_df = pd.DataFrame({"Metric": ["MAE", "RMSE", "R²"],
                        "Value": [mae, rmse, r2]})
display(eval_df)

# Scatter plot
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, s=65, color="teal")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.title("Predicted vs Actual")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 9. Feature Importance (absolute coefficients)
# ------------------------------------------------------------------
ohe = model.named_steps["prep"].named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(cat_features)
all_features = list(cat_names) + num_features
coefs = pd.Series(model.named_steps["reg"].coef_, index=all_features)

top20 = coefs.abs().sort_values(ascending=False).head(20)

plt.figure(figsize=(6, 8))
sns.barplot(x=top20.values, y=top20.index, palette="viridis")
plt.title("Top 20 |β| (Ridge)")
plt.xlabel("|Coefficient|")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 10. Persist Model
# ------------------------------------------------------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_file = f"traffic_ridge_{timestamp}.pkl"
joblib.dump(model, model_file)

print("\n" + "="*60)
print("Step 8 – Model Saved")
print("="*60)
print("File:", model_file)
print("\n🎉  All done!")