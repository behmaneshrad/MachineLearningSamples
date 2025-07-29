import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Loading the dataset
print("Step 1: Loading and preprocessing the dataset")
data = pd.read_excel('dataset/ENB2012_data.xlsx')
print("Dataset Info:")
print(data.info())
print("\nFirst few rows of the dataset:")
print(data.head())

# Preparing features and targets
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']].values
y = data[['Y1', 'Y2']].values

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Initializing lists to store metrics
degrees = [1, 2, 3, 4]
metrics = {'Y1': {'r2': [], 'mse': []}, 'Y2': {'r2': [], 'mse': []}}

# Performing polynomial regression for different degrees
for degree in degrees:
    print(f"\nStep 2: Polynomial Regression with Degree {degree}")
    
    # Creating polynomial regression pipeline
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    # Fitting the model
    polyreg.fit(X_train, y_train)
    
    # Getting model parameters
    linear_reg = polyreg.named_steps['linearregression']
    print(f"Model coefficients for degree {degree}:")
    print("Coefficients shape:", linear_reg.coef_.shape)
    print("Intercept:", linear_reg.intercept_)
    
    # Making predictions
    y_pred = polyreg.predict(X_test)
    
    # Calculating metrics
    r2_y1 = r2_score(y_test[:, 0], y_pred[:, 0])
    mse_y1 = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    r2_y2 = r2_score(y_test[:, 1], y_pred[:, 1])
    mse_y2 = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    
    metrics['Y1']['r2'].append(r2_y1)
    metrics['Y1']['mse'].append(mse_y1)
    metrics['Y2']['r2'].append(r2_y2)
    metrics['Y2']['mse'].append(mse_y2)
    
    print(f"Y1 (Heating Load) - R²: {r2_y1:.4f}, MSE: {mse_y1:.4f}")
    print(f"Y2 (Cooling Load) - R²: {r2_y2:.4f}, MSE: {mse_y2:.4f}")
    
    # Plotting actual vs predicted values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], 
             [y_test[:, 0].min(), y_test[:, 0].max()], 'r--', lw=2)
    plt.xlabel('Actual Y1 (Heating Load)')
    plt.ylabel('Predicted Y1')
    plt.title(f'Y1: Degree {degree} (R²: {r2_y1:.4f})')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], 
             [y_test[:, 1].min(), y_test[:, 1].max()], 'r--', lw=2)
    plt.xlabel('Actual Y2 (Cooling Load)')
    plt.ylabel('Predicted Y2')
    plt.title(f'Y2: Degree {degree} (R²: {r2_y2:.4f})')
    
    plt.tight_layout()
    plt.savefig(f'actual_vs_predicted_degree_{degree}.png')
    plt.close()

# Plotting performance metrics
print("\nStep 3: Plotting performance metrics")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(degrees, metrics['Y1']['r2'], 'b-o', label='R²')
plt.plot(degrees, metrics['Y1']['mse'], 'r-o', label='MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Metric Value')
plt.title('Y1 (Heating Load) Performance')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(degrees, metrics['Y2']['r2'], 'b-o', label='R²')
plt.plot(degrees, metrics['Y2']['mse'], 'r-o', label='MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Metric Value')
plt.title('Y2 (Cooling Load) Performance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('performance_metrics.png')
plt.close()

print("\nStep 4: Analysis Complete")
print("Performance metrics saved in 'performance_metrics.png'")
print("Actual vs Predicted plots saved as 'actual_vs_predicted_degree_X.png'")