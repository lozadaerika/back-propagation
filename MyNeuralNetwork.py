import numpy as np

# Neural Network class
class MyNeuralNetwork:
  def __init__(self, layers):
    self.L = len(layers)    # number of layers
    self.n = layers.copy()  # number of neurons in each layer

    self.xi = []            # node values
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay]))

    self.w = []             # edge weights
    self.w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.w.append(np.zeros((layers[lay], layers[lay - 1])))
  
  def fit(X, y):
    """Train the network using backpropagation""" 
    """X-> array (n_samples,n_features), which holds the training samples represented as floating point feature vectors"""
    """y->a vector of size (n_samples), which holds the target values (class labels) for the training samples"""
  
  def  predict(X):
    """Predict output from input X"""
    """X-> array of size (n_samples,n_features) that contains the samples. This method returns a vector with the predicted values for all the input samples"""
  
  def loss_epochs():
    """Calculate and return average loss over epochs""" 
    """returns 2 arrays of size (n_epochs, 2) that contain the evolution of the training error and the validation error for each of the epochs of the system, so this information can be plotted"""
    

# layers include input layer + hidden layers + output layer
layers = [4, 9, 5, 1]
nn = MyNeuralNetwork(layers)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")
