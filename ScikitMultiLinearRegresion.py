import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Read input and target data from file

df= pd.read_csv('A1-turbine.txt',sep='	')
input_data = df['#height_over_sea_level']
target_data = df['power_of_hydroelectrical_turbine']

input_columns=4
output_column=5
test_percentage=0.15
validation_percentage=0.25
activation_function='sigmoid'
#learning_rate=0.01
#momentum=0.9
#epochs=1000

# Split the data into training and test sets
input_train_val, input_test, target_train_val, target_test = train_test_split(input_data, target_data, test_size=test_percentage, random_state=42)
# Split the training and validation sets
input_train, input_val, target_train, target_val = train_test_split(input_train_val, target_train_val, test_size=validation_percentage, random_state=42)

# Scikit-learn's multi-linear regression
model = LinearRegression()

# Fit the model
model.fit(input_train, target_train)

# Evaluate model on the validation set
validation_predictions = model.predict(input_val)
print("Validation Predictions (Scikit-learn Linear Regression):", validation_predictions)

# Evaluate model on the test set
test_predictions = model.predict(input_test)
print("Test Predictions (Scikit-learn Linear Regression):", test_predictions)

validation_loss = np.mean((validation_predictions - target_val) ** 2)
test_loss = np.mean((test_predictions - target_test) ** 2)
print("Validation Loss (Scikit-learn Linear Regression):", validation_loss)
print("Test Loss (Scikit-learn Linear Regression):", test_loss)

plt.plot(test_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
