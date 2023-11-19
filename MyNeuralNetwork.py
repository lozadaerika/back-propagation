import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

    print('layers',layers)
    print('L',self.L)
    print('n',self.n)

    # Variables initialization
    # fields
    self.h = [np.zeros((1, size)) for size in layers]
    # activations
    self.xi = [np.zeros((1, size)) for size in layers]  
    # weights
    #self.w = [np.zeros((layers[i], layers[i+1])) for i in range(self.L - 1)]  
    self.w = [np.random.randn(layers[i], layers[i+1]) for i in range(self.L - 1)] #standard normal distribution randn
    # thresholds
    #self.theta = [np.zeros((1, size)) for size in layers[1:]]  
    self.theta = [np.random.randn(1, size) for size in layers[1:]] #standard normal distribution randn
    # errors propagation
    self.delta = [np.zeros((1, size)) for size in layers] 
    # weights changes
    self.d_w = [np.zeros((layers[i], layers[i+1])) for i in range(self.L - 1)]  
    # weights changes
    self.d_theta = [np.zeros((1, size)) for size in layers[1:]]  
    # previous weights changes used for momentum
    self.d_w_prev = [np.zeros((layers[i], layers[i+1])) for i in range(self.L - 1)]  
    # previous thresholds changes used for momentum
    self.d_theta_prev = [np.zeros((1, size)) for size in layers[1:]]  

    print(f"h: {self.h}, xi: {self.xi}, w: {self.w}, theta: {self.theta}, delta: {self.delta}, d_w: {self.d_w}, d_theta: {self.d_theta}, d_theta_prev: {self.d_theta_prev}")

    self.train_loss_history = []
    self.validation_loss_history = []

    # node values 
    self.xi = []            
    for layer in range(self.L):
      self.xi.append(np.zeros(layers[layer]))

    # edge weights
    self.w = []             
    self.w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.w.append(np.zeros((layers[lay], layers[lay - 1])))
  
  def feed_forward(self, input):
    self.xi[0] = input
    for i in range(self.L - 1):
      self.h[i + 1] = np.dot(self.xi[i], self.w[i]) + self.theta[i]
      self.xi[i + 1] = self.activation(self.h[i + 1])
    return self.xi[-1]

  def activation(self, x):
        if self.activation_function == 'sigmoid':
            return (1 / (1 + np.exp(-x)))
        else:
            raise ValueError("Activation function not supported")
  
  def activation_derivative(self, x):
        if self.activation_function == 'sigmoid':
            return x * (1 - x)
        else:
            raise ValueError("Activation function derivative not supported")
 
  def backward(self, targets):
        self.delta[-1] = (targets - self.xi[-1]) * self.activation_derivative(self.xi[-1])
        for i in range(self.L - 2, 0, -1):
            self.delta[i] = self.delta[i + 1].dot(self.w[i].T) * self.activation_derivative(self.xi[i])
  
  def update_weights_thresholds(self,x):
        for i in range(self.L - 1):
            self.d_w[i] = self.xi[i].T.dot(self.delta[i + 1]) * self.learning_rate
            self.d_theta[i] = np.sum(self.delta[i + 1], axis=0, keepdims=True) * self.learning_rate

        for i in range(self.L - 1):
            self.w[i] += self.d_w[i] + self.momentum * self.d_w_prev[i]
            self.theta[i] += self.d_theta[i] + self.momentum * self.d_theta_prev[i]
            self.d_w_prev[i] = self.d_w[i]
            self.d_theta_prev[i] = self.d_theta[i]
  

  def fit(self,X, y):
    """Train the network using backpropagation""" 
    """X-> array (n_samples,n_features), which holds the training samples represented as floating point feature vectors"""
    """y->a vector of size (n_samples), which holds the target values (class labels) for the training samples"""

    # Split the training and validation sets
    training_patterns, input_val, target_train, target_val = train_test_split(X, y, test_size=self.validation_percentage, random_state=42)

    for epoch in range(self.epochs):
        for pat in range(len(training_patterns)):
          #random pattern 
          random_index = np.random.randint(len(training_patterns))
          x_mu, z_mu = training_patterns[random_index], target_train[random_index]

          # Feed-forward propagation of pattern xµ to obtain the output o(xµ)
          output_train = self.feed_forward(x_mu.reshape(1, -1))
          # Back-propagate the error for this pattern
          self.backward(z_mu.reshape(1, -1))

          # Update the weights and thresholds and prevs
          for i in range(self.L - 1):
              self.w[i] += self.d_w[i] + self.momentum * self.d_w_prev[i]
              self.theta[i] += self.d_theta[i] + self.momentum * self.d_theta_prev[i]
              self.d_w_prev[i] = self.d_w[i]
              self.d_theta_prev[i] = self.d_theta[i]
        
        # Feed−forward patterns and calculate their prediction quadratic error
        train_predictions = np.array([self.feed_forward(x.reshape(1, -1)) for x in training_patterns])
        training_error = np.mean(np.square(target_train - train_predictions))
        val_predictions = np.array([self.feed_forward(x.reshape(1, -1)) for x in input_val])
        validation_error = np.mean(np.square(target_val - val_predictions))

        self.train_loss_history.append(training_error)
        self.validation_loss_history.append(validation_error)
  
  def  predict(self,X):
    """Predict output from input X"""
    """X-> array of size (n_samples,n_features) that contains the samples. This method returns a vector with the predicted values for all the input samples"""
    return  np.array([self.forward(x.reshape(1, -1)) for x in X])

  def loss_epochs(self):
    """Calculate and return average loss over epochs""" 
    """returns 2 arrays of size (n_epochs, 2) that contain the evolution of the training error and the validation error for each of the epochs of the system, so this information can be plotted"""
    return np.array(self.train_loss_history), np.array(self.validation_loss_history)


file_content =  np.loadtxt('A1-synthetic.txt')

input_columns=4
output_column=5
test_percentage=0.15
validation_percentage=0.25
activation_function='sigmoid'
learning_rate=0.01
momentum=0.9
epochs=1000

input_data = file_content[:, :input_columns]
target_data = file_content[output_column]

# Split the data into training and test sets
validation_input_data, input_test, validation_target_data, target_test = train_test_split(input_data, target_data, test_size=test_percentage, random_state=42)

# layers include input layer + hidden layers + output layer
layers = [input_data.shape[1], 9, 5, 1]
nn = MyNeuralNetwork(layers, epochs=epochs, learning_rate=learning_rate, momentum=momentum,activation_function=activation_function,validation_percentage=validation_percentage)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")
print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")
print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")

nn.fit(validation_input_data, validation_target_data)

# Predict using the trained network
predictions = nn.predict(validation_input_data)
print("Training Prediction:", predictions)

#Plot training and validation loss history
training_loss, validation_loss = nn.loss_epochs()
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Predict using the test set
test_prediction = nn.predict(input_test)
print("Test Prediction:", test_prediction)
