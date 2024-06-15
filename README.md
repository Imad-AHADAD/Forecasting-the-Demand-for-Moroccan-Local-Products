# Trade Value Forecasting Project

This project aims to forecast the trade values of different products using various machine learning and statistical models. The models used include Random Forest Regressor, Linear Regression, Support Vector Regressor, LSTM, and ARIMA.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Data](#data)
- [Results](#results)
- [Contributing](#contributing)
- [Technologies Used](#technologies Used)

## Installation

To set up the environment:
 Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Loading the Model and Making Predictions

```python
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the ARIMA model
with open('arima_model.pkl', 'rb') as file:
    model_fit = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load your new dataset for prediction
new_df = pd.read_csv('new_data.csv')

# Preprocess the new data
new_df = new_df.sort_values(by='Year')
target = 'Trade Value Growth Value'
new_df[target] = scaler.transform(new_df[[target]])

# Make a prediction for the next time step
predicted_value = model_fit.forecast(steps=1)
predicted_value_inverse = scaler.inverse_transform(predicted_value.reshape(-1, 1))

print(f'Predicted Value: {predicted_value_inverse[0][0]}')



## Models
Random Forest Regressor
Used for feature importance and initial predictions.
Linear Regression
A basic regression model for baseline comparisons.
Support Vector Regressor
Used for capturing complex relationships in the data.
LSTM
A neural network model used for time series forecasting.
ARIMA
A statistical model used for time series analysis and forecasting.
Data
The dataset used for this project includes trade values of various products over multiple years. The main features used are HS4 ID and Year, and the target variable is Trade Value Growth Value.

Results
The performance of each model is evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) metrics.



## Technologies Used
Python: The primary programming language used for data manipulation, modeling, and analysis.
Pandas: For data manipulation and analysis, providing data structures like DataFrames.
NumPy: For numerical operations and handling arrays.
Seaborn: For data visualization, especially statistical plots.
Matplotlib: For creating static, animated, and interactive visualizations in Python.
Scikit-Learn: For machine learning algorithms, preprocessing, and evaluation metrics.
RandomForestRegressor: For feature importance and regression tasks.
LinearRegression: For baseline regression analysis.
SVR (Support Vector Regressor): For regression analysis.
OneHotEncoder: For encoding categorical features.
MinMaxScaler: For feature scaling.
TensorFlow (Keras): For building and training the LSTM neural network model.
Sequential
LSTM
Dense
Statsmodels: For statistical modeling and time series analysis.
ARIMA: For time series forecasting.
Pickle: For saving and loading machine learning models and preprocessing objects.
Google Colab: For writing and running the notebook, as well as leveraging its computational resources.
Additional Libraries for Utility and Visualization
Seaborn: For enhanced data visualizations.
Matplotlib: For plotting and visualizing data.
IPython: For an enhanced interactive Python environment, particularly in Jupyter notebooks.
Jupyter Notebook: For creating and sharing documents that contain live code, equations, visualizations, and narrative text.
Project Structure
Data Loading: Reading CSV files using pandas.
Data Preprocessing: Handling missing values, feature scaling using MinMaxScaler, and encoding categorical variables using OneHotEncoder.
Exploratory Data Analysis (EDA): Visualizing data distributions, correlations, and trends using seaborn and matplotlib.
Feature Selection: Using RandomForestRegressor for determining the importance of features.
Model Training and Evaluation: Training multiple models (Random Forest, Linear Regression, SVR, LSTM, ARIMA) and evaluating their performance using MSE, MAE, and R² metrics.
Model Saving and Loading: Saving trained models and scalers using pickle for future use.
Forecasting: Using the ARIMA model for time series forecasting.
