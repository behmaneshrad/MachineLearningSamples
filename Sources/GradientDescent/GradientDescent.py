import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Loading and Preparing the Data
fileName='boston_house_prices.csv'
dataframe = pd.read_csv(fileName)
print(dataframe[['LSTAT', 'MEDV']].head())

#Normalizing the LSTAT and MEDV Columns
scaler = StandardScaler()
dataframe['LSTAT'] = scaler.fit_transform(dataframe[['LSTAT']])
print(dataframe[['LSTAT', 'MEDV']].head())


#Feature Preparation and Adding Bias
X = dataframe['LSTAT'].values.reshape(-1, 1)
y = dataframe['MEDV'].values

ones = np.ones((X.shape[0], 1))
X_b = np.hstack([ones, X])  
print(X_b[:5])

#Initializing Parameters and Gradient Descent Settings
theta = np.zeros(X_b.shape[1]) 

learning_rate = 0.01
epochs = 1000

#Running Gradient Descent and Tracking Cost History
m = len(y)
cost_history = []

for epoch in range(epochs):
    yHat = X_b.dot(theta)
    error = yHat - y  

    cost = np.sum(error**2)
    cost_history.append(cost)

    gradients = (1 / m) * X_b.T.dot(error)
    theta = theta - learning_rate * gradients

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Cost {cost:.4f}")

#Plotting the SSE Trend During Gradient Descent
plt.figure(figsize=(5, 3))
plt.plot(range(epochs), cost_history, color='darkred')
plt.xlabel('Epoch')
plt.ylabel('SSE (Sum of Squared Errors)')
plt.title('SSE Over Gradient Descent Epochs')
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"SSE: {cost_history[-1]:.4f}")
print(f"Final parameters (theta): {theta}")

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, X_b.dot(theta), color='red', linewidth=2, label='Regression Line')
plt.xlabel('LSTAT (scaled)')
plt.ylabel('MEDV (scaled)')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()