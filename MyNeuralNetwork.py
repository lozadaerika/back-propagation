import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math

# Neural Network class
class MyNeuralNetwork:

  def __init__(self, layers, epochs=1000, learning_rate=0.01,momentum=0.7,activation_function='sigmoid', validation_percentage=0.8):
    # number of layers
    self.L = len(layers) 
    # number of neurons for layer   
    self.n = layers.copy()  
    self.activation_function = activation_function
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.epochs= epochs
    self.validation_percentage=validation_percentage


    # Variables initialization
    # fields
    self.h = [np.zeros((size)) for size in layers]
    # activations
    self.xi = [np.zeros((size)) for size in layers]
    # weights
    self.w = [np.zeros((1, 1))] + [np.zeros((layers[i], layers[i-1])) for i in range(1,self.L)]  
    # thresholds
    self.theta = [np.zeros((size)) for size in layers]  
    # errors propagation
    self.delta = [np.zeros((size)) for size in layers] 
    # weights changes
    self.d_w = [np.zeros((1, 1))] + [np.zeros((layers[i], layers[i-1])) for i in range(1,self.L)]  
    # weights changes
    self.d_theta = [np.zeros(( size)) for size in layers]  
    # previous weights changes used for momentum
    self.d_w_prev = [np.zeros((1, 1))] + [np.zeros((layers[i], layers[i-1])) for i in range(1,self.L)]  
    # previous thresholds changes used for momentum
    self.d_theta_prev = [np.zeros(( size)) for size in layers]

    #print(f"h: {self.h}, xi: {self.xi}, w: {self.w}, theta: {self.theta}, delta: {self.delta}, d_w: {self.d_w}, d_theta: {self.d_theta}, d_theta_prev: {self.d_theta_prev}")
    self.train_loss_history = []
    self.validation_loss_history = []
  
    #Weights
    for lay in range(1,self.L):
      for neuron in range(self.n[lay]):
        for j in range(self.n[lay - 1]):
            self.w[lay][neuron][j] = np.random.uniform(-1, 1)
            self.w[lay][neuron][j] = np.random.uniform(-1, 1) # eliminar
            self.d_w_prev[lay][neuron][j] = 0

    #Thresholds
    for lay in range(1,self.L-1):
        for neuron in range(self.n[lay]):
            self.theta[lay][neuron] = np.random.uniform(-1, 1)
            self.d_theta_prev[lay][neuron] = 0
  
  def feed_forward(self, input):
    # Input layer
    for neuron in range(self.n[0]):
      self.xi[0][neuron] = input[neuron]

    # Hidden layers and output layer
    for layer in range(1, self.L):
      for neuron in range(self.n[layer]):
        self.h[layer][neuron] = 0
        for j in range(self.n[layer - 1]):
          #self.h[layer][neuron] += self.w[layer][neuron][j] * self.xi[layer - 1][j]
           val=self.valuesValidation(self.w[layer][neuron][j],self.xi[layer - 1][j])
           self.h[layer][neuron] += val
        
        self.h[layer][neuron] -= self.theta[layer][neuron]     
        self.xi[layer][neuron] = self.activation(self.h[layer][neuron])
    
    return self.xi[self.L - 1][0]
  
  def valuesValidation(self,value1, value2):
    result=0
    res = value1*value2
    if(math.isnan(value1) or math.isinf(value1)):
       result=0
    if(math.isnan(value2) or math.isinf(value2)):
       result=0
    if(math.isnan(value1*value2) or math.isinf(value1*value2)):
       result=0
    if(math.isnan(res) or math.isinf(res)):
       result=0
    return result

  def activation(self, x):
        if self.activation_function == 'sigmoid':
            return (1 / (1 + np.exp(-x)))
        elif self.activation_function == 'relu':
            return np.maximum(0.01, x)
        elif self.activation_function == 'linear':
            return x
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Activation function not supported")
  
  def activation_derivative(self, x):
        if self.activation_function == 'sigmoid':
            return self.activation(x) * (1 - self.activation(x))
        elif self.activation_function == 'relu':
            return np.where(x > 0, 1, 0.01)
        elif self.activation_function == 'linear':
            return np.ones_like(x)
        elif self.activation_function == 'tanh':
            return 1 - np.power(np.tanh(x), 2)
        else:
            raise ValueError("Activation function derivative not supported")
 
  def backward(self, targets):
    #The error term delta in the output layer
    for neuron in range(self.n[self.L - 1]):
      #Difference between the actual output and the target output
      diff=(self.xi[self.L - 1][neuron] - targets)
      self.delta[self.L - 1][neuron] = self.activation_derivative(self.h[self.L - 1][neuron]) * diff
        
    for layer in range(self.L - 2, 0, -1):  
      for neuron in range(self.n[layer]):
        self.delta[layer][neuron] = 0
        for j in range(self.n[layer + 1]):
          self.delta[layer][neuron] += self.delta[layer + 1][j] * self.w[layer + 1][j][neuron]
          self.delta[layer][neuron] *= self.activation_derivative(self.h[layer][neuron])
  

  def update_weights_thresholds(self):
    for lay in range(self.L - 1, 0, -1):
        for neuron in range(self.n[lay]):
            for j in range(self.n[lay - 1]):
                self.update_weights(lay, neuron, j)
            self.update_thresholds(lay, neuron)
    
  def update_weights(self, lay, neuron, j):
    self.d_w[lay][neuron][j] = -self.learning_rate * self.delta[lay][neuron] * self.xi[lay - 1][j] + self.momentum * self.d_w_prev[lay][neuron][j]
    self.d_w_prev[lay][neuron][j] = self.d_w[lay][neuron][j]
    self.w[lay][neuron][j] += self.d_w[lay][neuron][j]

  def update_thresholds(self, lay, neuron):
    self.d_theta[lay][neuron] = self.learning_rate * self.delta[lay][neuron] + self.momentum * self.d_theta_prev[lay][neuron]
    self.d_theta_prev[lay][neuron] = self.d_theta[lay][neuron]
    self.theta[lay][neuron] += self.d_theta[lay][neuron]

  def fit(self,X, y):
    """Train the network using backpropagation""" 
    """X-> array (n_samples,n_features), which holds the training samples represented as floating point feature vectors"""
    """y->a vector of size (n_samples), which holds the target values (class labels) for the training samples"""

    # Split the training and validation sets
    training_patterns, input_val, target_train, target_val = train_test_split(X, y, test_size=self.validation_percentage, random_state=42)

    for epoch in range(self.epochs):
        training_used = set()
        for pat in range(len(training_patterns)):
           #random pattern 
          random_index = np.random.randint(len(training_patterns))

          while random_index in training_used:
              random_index = np.random.randint(0, len(training_patterns))
          training_used.add(random_index)

          x_mu, z_mu = training_patterns[random_index], target_train[random_index]

          # Feed-forward propagation of pattern xµ to obtain the output o(xµ)
          output_train = self.feed_forward(x_mu)
          # Back-propagate the error for this pattern
          self.backward(z_mu)
          self.update_weights_thresholds()
        
        # Feed−forward patterns and calculate their prediction quadratic error
        train_predictions = np.array([self.feed_forward(x) for x in training_patterns])
        training_error = np.mean(np.square(target_train - train_predictions))
        val_predictions = np.array([self.feed_forward(x) for x in input_val])
        validation_error = np.mean(np.square(target_val - val_predictions))

        print("Epoch:",epoch,"Validation Loss:",validation_error,"Training Loss:",training_error)

        self.train_loss_history.append((epoch,training_error))
        self.validation_loss_history.append((epoch,validation_error))
  
  def  predict(self,X):
    """Predict output from input X"""
    """X-> array of size (n_samples,n_features) that contains the samples. This method returns a vector with the predicted values for all the input samples"""
    return  np.array([self.feed_forward(x) for x in X])

  def loss_epochs(self):
    """Calculate and return average loss over epochs""" 
    """returns 2 arrays of size (n_epochs, 2) that contain the evolution of the training error and the validation error for each of the epochs of the system, so this information can be plotted"""
    return np.array(self.train_loss_history), np.array(self.validation_loss_history)

  def calculate_mape(self,y_real, y_pred):
    return np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    

fileName='A1-synthetic/A1-synthetic-normalized.csv'
#fileName='A1-turbine/A1-turbine-normalized.csv'

file_content= pd.read_csv(fileName,delimiter=',')

test_percentage=0.20
validation_percentage=0.20
#activation_function='sigmoid'
#activation_function='relu'
activation_function='linear'
#activation_function='tanh'
momentum=0.3
learning_rate=0.3
epochs=50
layers = [4, 9, 5, 1] 

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

