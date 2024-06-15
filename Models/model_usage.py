import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from statsmodels.tsa.arima.model import ARIMA

# Load the ARIMA model
with open('arima_model.pkl', 'rb') as file:
    model_fit = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load your new dataset for prediction
# new_df = pd.read_csv('new_data.csv')  # Example of loading new data

# Assume new_df is already loaded as shown in the image

# Preprocess the new data
# Sort by Year to maintain the temporal order
new_df = new_df.sort_values(by='Year')

# Select the target
target = 'Trade Value Growth Value'

# Normalize the target data
new_df[target] = scaler.transform(new_df[[target]])

# Use the last available data from the new_df to make a prediction
last_observation = new_df[target].values[-1]

# Make a prediction for the next time step
predicted_value = model_fit.forecast(steps=1)

# Inverse transform the prediction
predicted_value_inverse = scaler.inverse_transform(predicted_value.reshape(-1, 1))

print(f'Predicted Value: {predicted_value_inverse[0][0]}')
