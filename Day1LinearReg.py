# home_dataset.csv a data set that compares house size and house prices of properties recently sold in Brooklyn's Dumbo neighborhood
# has 59 pairs of data points, house_size and house_price

import numpy as np
import pandas as pd # for reading and analyzing data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split# training and testing model
# splits the generated data into training and testing sets
from sklearn.linear_model import LinearRegression

# load data from CSV file
data = pd.read_csv("home_dataset.csv")

house_sizes = data["HouseSize"].values
house_prices = data["HousePrice"].values

# create a scatter plot
"""plt.scatter(house_sizes, house_prices, marker = 'o', color = 'blue')
plt.title("House Prices vs Size")
plt.xlabel("House Size (sq.ft)")
plt.ylabel("House Price ($)")
plt.show()"""

x_train, x_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size=0.2, random_state=42)
# splits so 20% goes to testing the algorithm

x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

plt.scatter(x_test, y_test, marker='o', color='blue', label='Actual Prices')
plt.plot(x_test, predictions, color='red', linewidth=2, label='Predicted Prices')
plt.title('Dumbo Property Price Prediction with Linear Regression')
plt.xlabel('House Size (sq.ft)')
plt.ylabel('House Price (millions $)')
plt.legend()
plt.show()