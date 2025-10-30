 # ClimaSense: Machine Learning–Based Weather Prediction System

This project demonstrates a simple weather forecasting model using **Linear Regression** to predict temperature based on humidity levels. It uses Python libraries like `pandas`, `scikit-learn`, and `matplotlib` for data handling, model training, and visualization.

---

## 📌 Project Objective

To predict temperature from humidity using supervised machine learning (linear regression) and visualize the regression line on a scatter plot.

---

## 🔧 Tools & Libraries Used

- **Python**
- **pandas** – for data manipulation
- **matplotlib** – for data visualization
- **scikit-learn** – for building and training the regression model

---

## 📁 Dataset

A small dummy dataset is used with:
- Temperature values (in °C)
- Corresponding Humidity percentages (%)

```python
data = {
    'Temperature': [30, 32, 33, 35, 36, 37, 39, 40, 41, 43],
    'Humidity': [70, 65, 60, 55, 50, 45, 40, 38, 35, 30]
}

