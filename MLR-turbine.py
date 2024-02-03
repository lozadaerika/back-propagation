import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

def calculate_mape(y_real, y_pred):
    return np.mean(np.abs((y_real - y_pred) / y_real)) * 100

# Load dataset
file_path = 'A1-turbine/A1-turbine-normalized.csv'
fileName="results/MLR/"+file_path.split("/")[1].split(".")[0]
label="turbine"

df = pd.read_csv(file_path, delimiter=',')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(df.head())

#Separate train and test
X_train, X_test, y_train, y_test  = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42
)

# Linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

mape = calculate_mape(y_test, y_pred)
print("Real values:", y_test)
print("Test Prediction:", y_pred)
print(f'MAPE: {mape:.2f}%')

plt.scatter(y_test,y_pred, label='Comparation')
plt.xlabel('Real values')
plt.ylabel('Test Prediction')
plt.legend()
plt.savefig(fileName+"-comparation.png")
plt.show()


with open(fileName+"-output.txt", 'w') as file:      
    sys.stdout = file
    print("Coefficients Separable: ", model.coef_)
    print("Intercept Separable:", model.intercept_)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Separable Error:", mse)

sys.stdout = sys.__stdout__

print("Coefficients Separable: ", model.coef_)
print("Intercept Separable:", model.intercept_)
print("Mean Squared Separable Error:", mse)
