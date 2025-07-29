import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the dataset
print("Step 1: Loading the dataset")
data = pd.read_csv('dataset/airfoil_self_noise.dat.txt', sep='\s+', header=None,
                   names=['frequency', 'angle_attack', 'chord_length', 'velocity', 
                          'displacement_thickness', 'sound_level'])
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Visualize initial data (scatter plot of frequency vs sound_level)
plt.figure(figsize=(10, 6))
plt.scatter(data['frequency'], data['sound_level'], color='blue', alpha=0.5)
plt.title('Frequency vs Sound Level (Raw Data)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Sound Level (dB)')
plt.grid(True)
plt.savefig('frequency_vs_sound_raw.png')
plt.show()

# Step 2: Preprocess the data
print("\nStep 2: Preprocessing the data")
X = data[['frequency', 'angle_attack', 'chord_length', 'velocity', 'displacement_thickness']]
y = data['sound_level']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaler Mean:", scaler.mean_)
print("Scaler Standard Deviation:", np.sqrt(scaler.var_))
print("\nFirst 5 rows of scaled features:")
print(X_scaled[:5])

# Visualize scaled feature (frequency) vs sound_level
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], y, color='green', alpha=0.5)
plt.title('Scaled Frequency vs Sound Level')
plt.xlabel('Scaled Frequency')
plt.ylabel('Sound Level (dB)')
plt.grid(True)
plt.savefig('scaled_frequency_vs_sound.png')
plt.show()

# Step 3: Create polynomial features
print("\nStep 3: Creating polynomial features")
degree = 2
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
print("Polynomial Feature Names:", poly.get_feature_names_out(X.columns))
print("Shape of polynomial features:", X_poly.shape)
print("\nFirst 5 rows of polynomial features:")
print(X_poly[:5])

# Step 4: Split the data
print("\nStep 4: Splitting the data into training and testing sets")
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
print("Training set shape (X_train, y_train):", X_train.shape, y_train.shape)
print("Testing set shape (X_test, y_test):", X_test.shape, y_test.shape)

# Step 5: Train the polynomial regression model
print("\nStep 5: Training the polynomial regression model")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Step 6: Make predictions
print("\nStep 6: Making predictions")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print("\nFirst 5 predictions on training set:", y_train_pred[:5])
print("First 5 actual values on training set:", y_train.values[:5])
print("First 5 predictions on test set:", y_test_pred[:5])
print("First 5 actual values on test set:", y_test.values[:5])

# Visualize predictions vs actual (using frequency for simplicity)
# Revert to original features for visualization
X_train_orig = scaler.inverse_transform(X_train[:, :5])  # First 5 columns are original features
X_test_orig = scaler.inverse_transform(X_test[:, :5])
plt.figure(figsize=(10, 6))
plt.scatter(X_train_orig[:, 0], y_train, color='blue', alpha=0.5, label='Training Data')
plt.scatter(X_test_orig[:, 0], y_test, color='red', alpha=0.5, label='Test Data')
plt.scatter(X_test_orig[:, 0], y_test_pred, color='green', alpha=0.5, label='Predictions')
plt.title('Frequency vs Sound Level with Predictions')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Sound Level (dB)')
plt.legend()
plt.grid(True)
plt.savefig('predictions_vs_actual.png')
plt.show()

# Step 7: Evaluate the model
print("\nStep 7: Evaluating the model")
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Training Metrics:")
print(f"MAE: {train_mae:.4f}")
print(f"MSE: {train_mse:.4f}")
print(f"R²: {train_r2:.4f}")
print("\nTesting Metrics:")
print(f"MAE: {test_mae:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"R²: {test_r2:.4f}")

# Visualize residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, y_test - y_test_pred, color='purple', alpha=0.5)
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residual Plot (Test Set)')
plt.xlabel('Predicted Sound Level (dB)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.grid(True)
plt.savefig('residual_plot.png')
plt.show()