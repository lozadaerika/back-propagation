import numpy as np

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
    self.w = [np.random.rand(layers[i], layers[i+1]) for i in range(self.L - 1)]  
    # thresholds
    self.theta = [np.zeros((1, size)) for size in layers[1:]]  
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
nn = MyNeuralNetwork(layers, epochs=1000, learning_rate=0.01, momentum=0.9,activation_function='sigmoid',validation_percentage=0.8)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")
