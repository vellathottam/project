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

# Train the model using all the data
model.fit(X, y)

# Predict the value at a given time in the future
future_time = 100 # Replace with the desired future time
future_value = model.predict([[future_time]])
print('Predicted value at time {}: {}'.format(future_time, future_value[0]))
