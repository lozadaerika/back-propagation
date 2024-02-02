import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.metrics import accuracy_score,mean_squared_error, r2_score

def calculate_mape(y_real, y_pred):
    return np.mean(np.abs((y_real - y_pred) / y_real)) * 100

# Load dataset
file_path = 'A1-turbine/A1-turbine-normalized.csv'

fileName="results/NeuralNetwork/"+file_path.split("/")[1].split(".")[0]

label="turbine"

# Load data into DataFrames
df = pd.read_csv(file_path, delimiter=',',header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the dataset into training and test sets with a 80:20 ratio
X_train, X_test, y_train, y_test  = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

epoch=50

# Keras model
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(4, activation='relu'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=epoch, batch_size=32, validation_data=(X_test, y_test),verbose=0)

y_pred = model.predict(X_test,verbose=0)

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
    sys.stdout = file  # Redirect stdout
    # Print the coefficients and intercept
    accuracy = accuracy_score(y_test, y_pred)
    print("Coefficients Separable: ", model.coef_)
    print("Intercept Separable:", model.intercept_)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Separable Error:", mse)
    print("R-squared Separable:", r2)
    print("Accuracy:", accuracy*100)

sys.stdout = sys.__stdout__

print("Coefficients Separable: ", model.coef_)
print("Intercept Separable:", model.intercept_)
print("Mean Squared Separable Error:", mse)
print("R-squared Separable:", r2)
print("Accuracy:", accuracy*100)