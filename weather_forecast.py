import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample dummy weather data
data = {
    'Temperature': [30, 32, 33, 35, 36, 37, 39, 40, 41, 43],
    'Humidity': [70, 65, 60, 55, 50, 45, 40, 38, 35, 30]
}

df = pd.DataFrame(data)

# Features and Labels
X = df[['Humidity']]
y = df['Temperature']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predicted = model.predict(X_test)

# Results
print("Actual Temperatures:", y_test.values)
print("Predicted Temperatures:", predicted)

# Visualization
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.title('Weather Forecasting using Linear Regression')
plt.legend()
plt.show()
