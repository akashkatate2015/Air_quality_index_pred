import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset (assuming city_day.csv is in the same directory)
df = pd.read_csv("/content/city_day.csv")

# Handling missing values by filling with column means (for numeric columns)
numeric_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'O3', 'Benzene', 'Toluene', 'Xylene']
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Mapping categorical variables ('City' and 'AQI_Bucket') to numeric values
city_mapping = {city: i for i, city in enumerate(df['City'].unique())}
aqi_bucket_mapping = {bucket: i for i, bucket in enumerate(df['AQI_Bucket'].unique())}
df['City'] = df['City'].map(city_mapping)
df['AQI_Bucket'] = df['AQI_Bucket'].map(aqi_bucket_mapping)

# Drop rows with NaN values in the target variable (AQI)
df = df.dropna(subset=['AQI'])

# Creating features and labels for machine learning
features = df[['City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'O3', 'Benzene', 'Toluene', 'Xylene']]
labels = df['AQI']

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=2)

# Train and evaluate RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print("RandomForestRegressor R-squared score:", r2_rf)
print("RandomForestRegressor Mean Squared Error:", mse_rf)

# Plotting predictions vs. actual values for RandomForestRegressor
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('RandomForestRegressor Predictions vs. Actual')
plt.show()

# Train and evaluate XGBoost Regressor
xg_reg = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=0)
xg_reg.fit(X_train, y_train)
y_pred_xg = xg_reg.predict(X_test)
r2_xg = r2_score(y_test, y_pred_xg)
mse_xg = mean_squared_error(y_test, y_pred_xg)
print("XGBoost R-squared score:", r2_xg)
print("XGBoost Mean Squared Error:", mse_xg)

# Plotting predictions vs. actual values for XGBoost Regressor
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xg, color='green', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('XGBoost Regressor Predictions vs. Actual')
plt.show()

# Train and evaluate Neural Network (using TensorFlow/Keras)
model_nn = Sequential([
    Dense(64, activation='relu', input_shape=(12,)),
    Dense(32, activation='relu'),
    Dense(1)
])
model_nn.compile(optimizer='adam', loss='mean_squared_error')
model_nn.fit(X_train, y_train, epochs=20, verbose=0)
y_pred_nn = model_nn.predict(X_test).flatten()
r2_nn = r2_score(y_test, y_pred_nn)
mse_nn = mean_squared_error(y_test, y_pred_nn)
print("Neural Network R-squared score:", r2_nn)
print("Neural Network Mean Squared Error:", mse_nn)

# Plotting predictions vs. actual values for Neural Network
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_nn, color='purple', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Neural Network Predictions vs. Actual')
plt.show() 
