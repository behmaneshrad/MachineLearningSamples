import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

# Setting up the environment for plots
plt.style.use('seaborn')
sns.set_palette("deep")

# 1. Loading and Preprocessing Data
print("=== Stage 1: Loading and Preprocessing Data ===")
# Define the path to the dataset
data_path = Path("dataset/Folds5x2_pp.xlsx")
# Load the first sheet of the Excel file
df = pd.read_excel(data_path, sheet_name='Sheet1')
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())

# Check for missing or invalid values
print("\nMissing Values:")
print(df.isnull().sum())

# Features and target
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

# 2. Exploratory Data Analysis (EDA)
print("\n=== Stage 2: Exploratory Data Analysis ===")
# Plotting feature distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Feature Distributions')
for i, col in enumerate(X.columns):
    sns.histplot(X[col], ax=axes[i//2, i%2], kde=True)
    axes[i//2, i%2].set_title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.show()
print("Saved feature distributions plot as 'feature_distributions.png'")

# Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.show()
print("Saved correlation matrix plot as 'correlation_matrix.png'")
print("\nCorrelation Matrix:")
print(df.corr())

# 3. Data Preparation
print("\n=== Stage 3: Data Preparation ===")
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nFeature Scaling Parameters (StandardScaler):")
print("Means:", scaler.mean_)
print("Standard Deviations:", scaler.scale_)

# 4. Polynomial Regression (Degree 2 and 3)
print("\n=== Stage 4: Polynomial Regression ===")
# Initialize results dictionary
results = {}

# Function to train and evaluate polynomial regression
def train_poly_model(degree):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    print(f"\nPolynomial Features (Degree {degree}) Names:")
    print(poly.get_feature_names_out(['AT', 'V', 'AP', 'RH']))
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nPolynomial Regression (Degree {degree}) Parameters:")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print(f"MSE: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual PE')
    plt.ylabel('Predicted PE')
    plt.title(f'Polynomial Regression (Degree {degree}) - Actual vs Predicted')
    plt.savefig(f'poly_degree_{degree}_actual_vs_predicted.png')
    plt.show()
    print(f"Saved actual vs predicted plot for degree {degree} as 'poly_degree_{degree}_actual_vs_predicted.png'")
    
    return {'mse': mse, 'r2': r2}

# Train polynomial models
results['poly_degree_2'] = train_poly_model(2)
results['poly_degree_3'] = train_poly_model(3)

# 5. SVR Model
print("\n=== Stage 5: SVR Model ===")
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train)
y_pred_svr = svr.predict(X_test_scaled)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)
print("\nSVR Parameters:")
print("Kernel:", svr.kernel)
print("C:", svr.C)
print("Epsilon:", svr.epsilon)
print(f"MSE: {mse_svr:.2f}")
print(f"R² Score: {r2_svr:.2f}")

# Plot actual vs predicted for SVR
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_svr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual PE')
plt.ylabel('Predicted PE')
plt.title('SVR - Actual vs Predicted')
plt.savefig('svr_actual_vs_predicted.png')
plt.show()
print("Saved SVR actual vs predicted plot as 'svr_actual_vs_predicted.png'")
results['svr'] = {'mse': mse_svr, 'r2': r2_svr}

# 6. Random Forest Model
print("\n=== Stage 6: Random Forest Model ===")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("\nRandom Forest Parameters:")
print("Number of Estimators:", rf.n_estimators)
print("Feature Importances:", rf.feature_importances_)
print(f"MSE: {mse_rf:.2f}")
print(f"R² Score: {r2_rf:.2f}")

# Plot actual vs predicted for Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual PE')
plt.ylabel('Predicted PE')
plt.title('Random Forest - Actual vs Predicted')
plt.savefig('rf_actual_vs_predicted.png')
plt.show()
print("Saved Random Forest actual vs predicted plot as 'rf_actual_vs_predicted.png'")
results['rf'] = {'mse': mse_rf, 'r2': r2_rf}

# 7. Model Comparison
print("\n=== Stage 7: Model Comparison ===")
print("\nModel Performance Summary:")
for model_name, metrics in results.items():
    print(f"{model_name}: MSE = {metrics['mse']:.2f}, R² = {metrics['r2']:.2f}")

# Plot model comparison
models = list(results.keys())
mses = [results[model]['mse'] for model in models]
r2s = [results[model]['r2'] for model in models]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.bar(models, mses)
ax1.set_title('Model Comparison - MSE')
ax1.set_ylabel('Mean Squared Error')
ax2.bar(models, r2s)
ax2.set_title('Model Comparison - R² Score')
ax2.set_ylabel('R² Score')
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()
print("Saved model comparison plot as 'model_comparison.png'")