import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# Load the data
df = pd.read_csv('trade_morocoo_exports_processed.csv')

# Load the LSTM model
from tensorflow.keras.models import load_model
model_lstm = load_model('lstm_model.h5')

# Load the Linear Regression model
with open('linear_regression_model.pkl', 'rb') as file:
    model_lr = pickle.load(file)

# Load the SVR model
with open('svr_model.pkl', 'rb') as file:
    model_svr = pickle.load(file)

# Load the ARIMA model
with open('arima_model.pkl', 'rb') as file:
    model_arima = pickle.load(file)

# Load the scaler
with open('feature_scaler_lstm.pkl', 'rb') as file:
    scaler_lstm = pickle.load(file)

# Inverse transform function for LSTM model
def inverse_transform_lstm(data, scaler):
    return scaler.inverse_transform(data.reshape(-1, 1))

# Function to evaluate models and return metrics
def evaluate_models(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

# Function to make predictions using LSTM model
def predict_lstm(X, model, scaler):
    predictions = model.predict(X)
    predictions_inverse = inverse_transform_lstm(predictions, scaler)
    return predictions_inverse

# Function to make predictions using Linear Regression model
def predict_lr(X, model):
    return model.predict(X)

# Function to make predictions using SVR model
def predict_svr(X, model):
    return model.predict(X)

# Function to make predictions using ARIMA model
def predict_arima(data, model, scaler, target='Trade Value'):
    # Inverse transform the target
    data[target] = scaler.inverse_transform(data[[target]])
    
    # Define order (p, d, q)
    p, d, q = 5, 1, 0  # This can be tuned
    model_arima = ARIMA(data[target], order=(p, d, q))
    model_fit = model_arima.fit()

    # Make predictions
    predictions = model_fit.forecast(steps=len(data))

    return predictions

# Predict using LSTM model
X_lstm = df.iloc[-5:].drop('Trade Value', axis=1).values.reshape((1, 5, len(df.columns)-1))
y_pred_lstm = predict_lstm(X_lstm, model_lstm, scaler_lstm)

# Predict using Linear Regression model
X_lr = df.iloc[-1][['HS4 ID', 'Year']].values.reshape(1, -1)
y_pred_lr = predict_lr(X_lr, model_lr)

# Predict using SVR model
y_pred_svr = predict_svr(X_lr, model_svr)

# Predict using ARIMA model
df_arima = df.copy()
predictions_arima = predict_arima(df_arima, model_arima, scaler, target='Trade Value')

# Calculate evaluation metrics for LSTM model
y_true_lstm = df.iloc[-1]['Trade Value']
mse_lstm, mae_lstm, r2_lstm = evaluate_models(y_true_lstm, y_pred_lstm)

# Calculate evaluation metrics for Linear Regression model
y_true_lr = df.iloc[-1]['Trade Value']
mse_lr, mae_lr, r2_lr = evaluate_models(y_true_lr, y_pred_lr)

# Calculate evaluation metrics for SVR model
y_true_svr = df.iloc[-1]['Trade Value']
mse_svr, mae_svr, r2_svr = evaluate_models(y_true_svr, y_pred_svr)

# Calculate evaluation metrics for ARIMA model
y_true_arima = df.iloc[-1]['Trade Value']
mse_arima = mean_squared_error(y_true_arima, predictions_arima)
mae_arima = mean_absolute_error(y_true_arima, predictions_arima)
r2_arima = r2_score(y_true_arima, predictions_arima)

# Print evaluation metrics
print(f'LSTM Model - MSE: {mse_lstm}, MAE: {mae_lstm}, R²: {r2_lstm}')
print(f'Linear Regression Model - MSE: {mse_lr}, MAE: {mae_lr}, R²: {r2_lr}')
print(f'SVR Model - MSE: {mse_svr}, MAE: {mae_svr}, R²: {r2_svr}')
print(f'ARIMA Model - MSE: {mse_arima}, MAE: {mae_arima}, R²: {r2_arima}')

# Formulate recommendations based on the results
recommendations = []

# Example recommendations based on model performance
if r2_lstm > 0.8:
    recommendations.append("Utiliser le modèle LSTM pour les prévisions à long terme.")
if r2_lr > 0.7:
    recommendations.append("Utiliser la régression linéaire pour les prévisions à court terme.")
if r2_svr > 0.7:
    recommendations.append("Utiliser le SVR pour les prévisions à court terme.")
if r2_arima > 0.6:
    recommendations.append("Utiliser le modèle ARIMA pour les prévisions à court terme.")

print("\nRecommandations Stratégiques:")
for i, recommendation in enumerate(recommendations):
    print(f"{i + 1}. {recommendation}")

