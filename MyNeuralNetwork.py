import numpy as np
from sklearn.model_selection import train_test_split

# Neural Network class
class MyNeuralNetwork:

  def __init__(self, layers, epochs=1000, learning_rate=0.01,momentum=0.7,activation_function='sigmoid', validation_percentage=0.8):
    self.L = len(layers)    # number of layers
    self.n = layers.copy()  # number of neurons in each layer
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
    print('h',self.h)
    print('xi',self.xi)
    print('w',self.w)
    print('theta',self.theta)
    print('delta',self.delta)
    print('d_w',self.d_w)
    print('d_theta',self.d_theta)
    print('d_theta_prev',self.d_theta_prev)

    self.xi = []            # node values
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay]))

    self.w = []             # edge weights
    self.w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.w.append(np.zeros((layers[lay], layers[lay - 1])))
  
  def feedForward(self, input):
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

    for epoch in range(self.epochs):
        for i in range(len(training_patterns)):
          ##
          
        # Store the errors in the history
        

  
  def  predict(X):
    """Predict output from input X"""
    """X-> array of size (n_samples,n_features) that contains the samples. This method returns a vector with the predicted values for all the input samples"""
  
  def loss_epochs():
    """Calculate and return average loss over epochs""" 
    """returns 2 arrays of size (n_epochs, 2) that contain the evolution of the training error and the validation error for each of the epochs of the system, so this information can be plotted"""
    

# layers include input layer + hidden layers + output layer
layers = [4, 9, 5, 1]
nn = MyNeuralNetwork(layers, epochs=1000, learning_rate=0.01, momentum=0.9,activation_function='sigmoid',validation_percentage=0.8)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")
