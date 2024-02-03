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

df = pd.read_csv(file_path, delimiter=',',header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test  = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42
)

epoch=50

epoch=50

# Keras model
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy','mean_squared_error'])

model.fit(X_train, y_train, epochs=epoch, batch_size=32,verbose=0)

y_pred = model.predict(X_test,verbose=0)

y_pred_aux=y_test.copy()
flat_list = [item for sublist in y_pred for item in sublist]
for i in range(len(y_test)):
    y_pred_aux[i]=flat_list[i]
y_pred = y_pred_aux


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

results = model.evaluate(X_test, y_test)

with open(fileName+"-output.txt", 'w') as file:      
    sys.stdout = file
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Separable Error:", results[2])

sys.stdout = sys.__stdout__

print("Mean Squared Separable Error:", results[2])