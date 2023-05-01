import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('time_series_data.csv')

# Create features and target arrays
X = np.array(data['time']).reshape(-1, 1)
y = np.array(data['value'])

# Initialize the linear regression model
model = LinearRegression()

# Train the model using the first 80% of the data
split_idx = int(0.8 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
model.fit(X_train, y_train)

# Test the model on the remaining 20% of the data
X_test, y_test = X[split_idx:], y[split_idx:]
y_pred = model.predict(X_test)

# Print the R^2 score of the model
r2_score = model.score(X_test, y_test)
print('R^2 score:', r2_score)
