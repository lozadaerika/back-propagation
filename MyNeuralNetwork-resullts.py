from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import sys

from MyNeuralNetwork import MyNeuralNetwork

#fileName='A1-synthetic/A1-synthetic-normalized.csv'
fileName='A1-turbine/A1-turbine-normalized.csv'
#fileName='A1-personalized/A1-energy-normalized.csv'

file_content= pd.read_csv(fileName,delimiter=',')

test_percentage=0.20
validation_percentage=0.20
#activation_function='sigmoid'
#activation_function='relu'
activation_function='relu'
#activation_function='tanh'
momentum=0.1
learning_rate=0.3
epochs=100
layers = [4, 3, 1] 

saveFileName=fileName.split("/")[1].split(".")[0]+"-af-"+activation_function+"-e-"+str(epochs)+"-m-"+str(momentum)+"-lr-"+str(learning_rate)+"-vp-"+str(validation_percentage)+"-tp-"+str(test_percentage)

input_data = file_content.iloc[:, :-1]
target_data = file_content.iloc[:, -1]

input_data=input_data.values.tolist()
target_data=target_data.values.tolist()

# Split the data into training and test sets
validation_input_data, input_test, validation_target_data, target_test = train_test_split(input_data, target_data, test_size=test_percentage, random_state=42)

# layers include input layer + hidden layers + output layer
nn = MyNeuralNetwork(layers, epochs=epochs, learning_rate=learning_rate, momentum=momentum,activation_function=activation_function,validation_percentage=validation_percentage)

nn.fit(validation_input_data, validation_target_data)

#Plot training and validation loss history
training_loss, validation_loss = nn.loss_epochs()
plt.scatter(training_loss[:,0],training_loss[:,1], label='Training Loss')
plt.scatter(validation_loss[:,0],validation_loss[:,1], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.savefig("results/"+saveFileName+"-error.png")
plt.show()

# Predict using the test set
test_prediction = nn.predict(input_test)
mape = nn.calculate_mape(target_test, test_prediction)

with open("results/"+saveFileName+"output.txt", 'w') as file:      
    sys.stdout = file
    print("Test percentage:",test_percentage)
    print("Validation percentage:",validation_percentage)
    print("Activation function:",activation_function)
    print("Learning rate:",learning_rate)
    print("Momentum:",momentum)
    print("Epochs:",epochs)
    print("Real values:", target_test)
    print("Test Prediction:", test_prediction)
    print(f'MAPE: {mape:.2f}%')

# Reset stdout to the original value
sys.stdout = sys.__stdout__

print("Real values:", target_test)
print("Test Prediction:", test_prediction)
print(f'MAPE: {mape:.2f}%')

plt.scatter(target_test,test_prediction, label='Comparation')
plt.xlabel('Real values')
plt.ylabel('Test Prediction')
plt.legend()
plt.savefig("results/"+saveFileName+"-comparation.png")
plt.show()

