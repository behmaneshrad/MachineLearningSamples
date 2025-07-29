import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Setting up plotting style
plt.style.use('seaborn')

# 1. Loading and preprocessing the dataset
print("=== Step 1: Loading and Preprocessing Data ===")
data = pd.read_csv('dataset/abalone.data', 
                   names=['Sex', 'Length', 'Diameter', 'Height', 
                         'Whole_weight', 'Shucked_weight', 'Viscera_weight', 
                         'Shell_weight', 'Rings'])

# Converting Sex to numerical (one-hot encoding)
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
print("Dataset Info:")
print(data.info())
print("\nFirst few rows of preprocessed data:")
print(data.head())

# Visualizing correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.show()

# 2. Preparing features and target
print("\n=== Step 2: Preparing Features and Target ===")
X = data.drop('Rings', axis=1)
y = data['Rings']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Visualizing distribution of target variable
plt.figure(figsize=(8, 6))
sns.histplot(y, bins=30, kde=True)
plt.title('Distribution of Abalone Rings (Age)')
plt.xlabel('Rings')
plt.ylabel('Count')
plt.show()

# 3. Standardizing features
print("\n=== Step 3: Standardizing Features ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaled features mean (train):", X_train_scaled.mean(axis=0).round(4))
print("Scaled features std (train):", X_train_scaled.std(axis=0).round(4))

# 4. Polynomial feature transformation
print("\n=== Step 4: Polynomial Feature Transformation ===")
degrees = [1, 2, 3]
poly_results = []

for degree in degrees:
    print(f"\nTrying polynomial degree: {degree}")
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    # Fitting linear regression
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Making predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    # Calculating metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    poly_results.append({
        'degree': degree,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    })
    
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    
    # Visualizing actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Rings')
    plt.ylabel('Predicted Rings')
    plt.title(f'Actual vs Predicted Rings (Degree {degree})')
    plt.tight_layout()
    plt.show()

# 5. PCA for dimensionality reduction
print("\n=== Step 5: PCA Analysis ===")
# Using degree 3 for PCA as it often shows better performance
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Applying PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_poly)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Visualizing explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Explained Variance Ratio')
plt.grid(True)
plt.tight_layout()
plt.show()

# Selecting components that explain 95% of variance
n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components explaining 95% variance: {n_components}")

# Applying PCA with selected components
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_poly)
X_test_pca = pca.transform(X_test_poly)

# Fitting model with PCA features
model_pca = LinearRegression()
model_pca.fit(X_train_pca, y_train)
y_train_pred_pca = model_pca.predict(X_train_pca)
y_test_pred_pca = model_pca.predict(X_test_pca)

# Calculating lelating PCA metrics
train_rmse_pca = np.sqrt(mean_squared_error(y_train, y_train_pred_pca))
test_rmse_pca = np.sqrt(mean_squared_error(y_test, y_test_pred_pca))
train_r2_pca = r2_score(y_train, y_train_pred_pca)
test_r2_pca = r2_score(y_test, y_test_pred_pca)

print("\n=== Step 6: PCA Model Results ===")
print(f"Train RMSE (PCA): {train_rmse_pca:.4f}")
print(f"Test RMSE (PCA): {test_rmse_pca:.4f}")
print(f"Train R2 (PCA): {train_r2_pca:.4f}")
print(f"Test R2 (PCA): {test_r2_pca:.4f}")

# Visualizing PCA results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred_pca, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Actual vs Predicted Rings (PCA)')
plt.tight_layout()
plt.show()

# Comparing polynomial degrees
print("\n=== Step 7: Model Comparison ===")
for result in poly_results:
    print(f"\nPolynomial Degree {result['degree']}:")
    print(f"Train RMSE: {result['train_rmse']:.4f}")
    print(f"Test RMSE: {result['test_rmse']:.4f}")
    print(f"Train R2: {result['train_r2']:.4f}")
    print(f"Test R2: {result['test_r2']:.4f}")

# Visualizing model performance comparison
degrees = [result['degree'] for result in poly_results]
test_rmse = [result['test_rmse'] for result in poly_results]
plt.figure(figsize=(8, 6))
plt.plot(degrees, test_rmse, 'bo-', label='Test RMSE')
plt.plot(degrees, [test_rmse_pca] * len(degrees), 'r--', label='Test RMSE (PCA)')
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE')
plt.title('Model Performance Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()