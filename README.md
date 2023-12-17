# PRODIGY_ML_1
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate a sample dataset (replace this with your actual dataset)
np.random.seed(42)
data_size = 100
square_footage = np.random.randint(1000, 3000, size=data_size)
num_bedrooms = np.random.randint(2, 5, size=data_size)
num_bathrooms = np.random.randint(1, 4, size=data_size)
prices = 50000 + 300 * square_footage + 20000 * num_bedrooms + 25000 * num_bathrooms + np.random.normal(0, 10000, size=data_size)

# Create a DataFrame
df = pd.DataFrame({
    'SquareFootage': square_footage,
    'NumBedrooms': num_bedrooms,
    'NumBathrooms': num_bathrooms,
    'Prices': prices
})

# Split the dataset into features (X) and target variable (y)
X = df[['SquareFootage', 'NumBedrooms', 'NumBathrooms']]
y = df['Prices']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize predictions vs. actual prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()
