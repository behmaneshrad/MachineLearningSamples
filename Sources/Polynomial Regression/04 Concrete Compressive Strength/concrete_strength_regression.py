import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
print("Step 1: Loading the dataset")
data = pd.read_excel('dataset/Concrete_Data.xls')
print("Dataset Info:")
print(data.info())
print("\nDataset Head:")
print(data.head())

# Define features and target
X = data.iloc[:, :-1].values  # All columns except the last one
y = data.iloc[:, -1].values   # Last column (Concrete compressive strength)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nStep 2: Data Splitting")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize lists to store results
degrees = [1, 2, 3]
train_mse = []
test_mse = []
cv_scores = []
ridge_mse = []

# Plotting setup
plt.style.use('seaborn')

# Step 3: Polynomial Regression with different degrees
for degree in degrees:
    print(f"\nStep 3.{degree}: Polynomial Regression (Degree {degree})")
    
    # Create polynomial regression pipeline
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    # Fit model
    polyreg.fit(X_train_scaled.reshape(-1, X_train.shape[1]), y_train)
    
    # Predictions
    y_train_pred = polyreg.predict(X_train_scaled)
    y_test_pred = polyreg.predict(X_test_scaled)
    
    # Calculate MSE
    train_mse.append(mean_squared_error(y_train, y_train_pred))
    test_mse.append(mean_squared_error(y_test, y_test_pred))
    
    # Cross-validation
    cv_score = cross_val_score(polyreg, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_scores.append(cv_score.mean())
    
    # Print parameters
    print(f"Polynomial Degree: {degree}")
    print(f"Training MSE: {train_mse[-1]:.4f}")
    print(f"Test MSE: {test_mse[-1]:.4f}")
    print(f"Cross-validation R2 score: {cv_score.mean():.4f} ± {cv_score.std():.4f}")
    
    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Compressive Strength (MPa)')
    plt.ylabel('Predicted Compressive Strength (MPa)')
    plt.title(f'Polynomial Regression (Degree {degree}) - Predictions vs Actual')
    plt.tight_layout()
    plt.savefig(f'poly_degree_{degree}_predictions.png')
    plt.close()

# Step 4: Ridge Regression with Polynomial Features
print("\nStep 4: Ridge Regression with Polynomial Features (Degree 2)")
polyreg_ridge = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), Ridge(alpha=1.0))
polyreg_ridge.fit(X_train_scaled, y_train)

# Predictions
y_train_pred_ridge = polyreg_ridge.predict(X_train_scaled)
y_test_pred_ridge = polyreg_ridge.predict(X_test_scaled)

# Calculate MSE
ridge_train_mse = mean_squared_error(y_train, y_train_pred_ridge)
ridge_test_mse = mean_squared_error(y_test, y_test_pred_ridge)
ridge_mse.append(ridge_test_mse)

# Cross-validation
ridge_cv_score = cross_val_score(polyreg_ridge, X_train_scaled, y_train, cv=5, scoring='r2')

# Print parameters
print(f"Ridge Regression (Degree 2) Parameters:")
print(f"Alpha: 1.0")
print(f"Training MSE: {ridge_train_mse:.4f}")
print(f"Test MSE: {ridge_test_mse:.4f}")
print(f"Cross-validation R2 score: {ridge_cv_score.mean():.4f} ± {ridge_cv_score.std():.4f}")

# Plot Ridge predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred_ridge, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Compressive Strength (MPa)')
plt.ylabel('Predicted Compressive Strength (MPa)')
plt.title('Ridge Regression (Degree 2) - Predictions vs Actual')
plt.tight_layout()
plt.savefig('ridge_degree_2_predictions.png')
plt.close()

# Step 5: Compare Models
print("\nStep 5: Model Comparison")
plt.figure(figsize=(12, 6))
plt.plot(degrees, train_mse, 'o-', label='Training MSE')
plt.plot(degrees, test_mse, 'o-', label='Test MSE')
plt.axhline(y=ridge_test_mse, color='g', linestyle='--', label='Ridge Test MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Model Performance Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Step 6: Feature Importance Analysis (for linear regression)
print("\nStep 6: Feature Importance Analysis (Linear Regression)")
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Get feature names
feature_names = data.columns[:-1]

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(feature_names, abs(linear_model.coef_))
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.title('Feature Importance (Linear Regression)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Print feature importance
print("Feature Importance (Absolute Coefficients):")
for feature, coef in zip(feature_names, abs(linear_model.coef_)):
    print(f"{feature}: {coef:.4f}")

# Save results summary
results = {
    'Degree': degrees + ['Ridge (2)'],
    'Train MSE': train_mse + [ridge_train_mse],
    'Test MSE': test_mse + [ridge_test_mse],
    'CV R2 Score': cv_scores + [ridge_cv_score.mean()]
}
results_df = pd.DataFrame(results)
results_df.to_csv('model_results.csv', index=False)
print("\nResults saved to 'model_results.csv'")
print(results_df)