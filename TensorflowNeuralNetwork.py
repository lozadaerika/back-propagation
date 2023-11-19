import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read input and target data from file
file_content =  np.loadtxt('A1-synthetic.txt')

input_columns=4
output_column=5
test_percentage=0.15
validation_percentage=0.25
activation_function='sigmoid'
#learning_rate=0.01
#momentum=0.9
#epochs=1000

input_data = file_content[:, :input_columns]
target_data = file_content[output_column]

# Split the data into training and test sets
input_train_val, input_test, target_train_val, target_test = train_test_split(input_data, target_data, test_size=test_percentage, random_state=42)

# Split the training and validation sets
input_train, input_val, target_train, target_val = train_test_split(input_train_val, target_train_val, test_size=validation_percentage, random_state=42)

# Build a simple neural network model using TensorFlow and Keras
#layers = [4, 9, 5, 1]
model = tf.keras.Sequential([
    tf.keras.layers.Dense(9, activation=activation_function, input_shape=(input_data.shape[1],)),
    tf.keras.layers.Dense(5, activation=activation_function), 
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
history = model.fit(input_train, target_train, validation_data=(input_val, target_val), epochs=100, verbose=1)

# Plot the loss history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Evaluate the model and predict on the validation set
validation_loss = model.evaluate(input_val, target_val, verbose=0)
print("Validation Loss:", validation_loss)
validation_predictions = model.predict(input_val)
print("Validation Predictions:", validation_predictions)

# Evaluate the model and predict on the test set
test_loss = model.evaluate(input_test, target_test, verbose=0)
print("Test Loss:", test_loss)
test_predictions = model.predict(input_test)
print("Test Predictions:", test_predictions)
