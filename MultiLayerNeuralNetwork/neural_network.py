import numpy as np


# Define activation functions and their derivatives
def sigmoid(x):
  """Sigmoid activation function."""
  return 1 / (1 + np.exp(-x))


def dsigmoid(a):
  """
  Derivative of the sigmoid.
  Note: Assumes a = sigmoid(x) has already been computed.
  """
  return a * (1 - a)


def tanh(x):
  """Hyperbolic tangent activation function."""
  return np.tanh(x)


def dtanh(a):
  """
  Derivative of tanh.
  Note: Assumes a = tanh(x) has already been computed.
  """
  return 1 - a ** 2


# Define the Neural Network class
class NeuralNetwork:
  def __init__(self, layer_sizes, activation='sigmoid', learning_rate=0.1):
    """
    Initialize the neural network.

    Parameters:
        layer_sizes (list): A list containing the number of neurons in each layer.
                            For example, [input_dim, hidden1, ..., output_dim].
        activation (str):  The activation function to use ('sigmoid' or 'tanh').
        learning_rate (float): Learning rate for gradient descent.
    """
    self.layer_sizes = layer_sizes
    self.learning_rate = learning_rate
    self.num_layers = len(layer_sizes)

    # Initialize weights and biases for each layer
    self.weights = []
    self.biases = []
    for i in range(self.num_layers - 1):
      # He initialization for weights
      weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
      self.weights.append(weight)
      bias = np.zeros((1, layer_sizes[i + 1]))
      self.biases.append(bias)

    # Select activation function and its derivative
    if activation == 'sigmoid':
      self.activation = sigmoid
      self.activation_derivative = dsigmoid
    elif activation == 'tanh':
      self.activation = tanh
      self.activation_derivative = dtanh
    else:
      raise ValueError("Unsupported activation function. Choose 'sigmoid' or 'tanh'.")

  def forward(self, X):
    """
    Perform forward propagation.

    Parameters:
        X (numpy.ndarray): Input data of shape (n_samples, n_features).

    Returns:
        activations (list): Activations for each layer.
        zs (list): The linear combinations (weighted inputs) for each layer.
    """
    activations = [X]
    zs = []  # Store weighted sums
    for w, b in zip(self.weights, self.biases):
      z = np.dot(activations[-1], w) + b
      zs.append(z)
      a = self.activation(z)
      activations.append(a)
    return activations, zs

  def backward(self, X, y, activations, zs):
    """
    Perform backpropagation to compute the gradients.

    Parameters:
        X (numpy.ndarray): Input data.
        y (numpy.ndarray): True labels.
        activations (list): List of activations from forward propagation.
        zs (list): List of weighted inputs from forward propagation.

    Returns:
        nabla_w (list): Gradients of weights.
        nabla_b (list): Gradients of biases.
    """
    m = y.shape[0]  # Number of samples

    # Initialize gradient lists with None
    nabla_w = [None] * len(self.weights)
    nabla_b = [None] * len(self.biases)

    # Compute error for output layer
    # Using mean squared error (MSE) loss: dC/da = (a - y)
    delta = (activations[-1] - y) * self.activation_derivative(activations[-1])
    nabla_w[-1] = np.dot(activations[-2].T, delta) / m
    nabla_b[-1] = np.sum(delta, axis=0, keepdims=True) / m

    # Backpropagate the error to earlier layers
    for l in range(2, self.num_layers):
      # l = 2 corresponds to second-last layer, etc.
      z = zs[-l]
      sp = self.activation_derivative(activations[-l])
      delta = np.dot(delta, self.weights[-l + 1].T) * sp
      nabla_w[-l] = np.dot(activations[-l - 1].T, delta) / m
      nabla_b[-l] = np.sum(delta, axis=0, keepdims=True) / m

    return nabla_w, nabla_b

  def update_parameters(self, nabla_w, nabla_b):
    """Update weights and biases using gradient descent."""
    for i in range(len(self.weights)):
      self.weights[i] -= self.learning_rate * nabla_w[i]
      self.biases[i] -= self.learning_rate * nabla_b[i]

  def train(self, X, y, epochs=1000):
    """
    Train the neural network.

    Parameters:
        X (numpy.ndarray): Training input data.
        y (numpy.ndarray): Training labels.
        epochs (int): Number of training iterations.
    """
    for epoch in range(epochs):
      # Forward propagation
      activations, zs = self.forward(X)
      # Compute cost (MSE)
      cost = np.mean((activations[-1] - y) ** 2)

      # Backpropagation
      nabla_w, nabla_b = self.backward(X, y, activations, zs)
      # Update parameters
      self.update_parameters(nabla_w, nabla_b)

      # Optionally print the cost every 100 epochs
      if epoch % 100 == 0:
        print(f"Epoch {epoch}: Cost = {cost}")

  def predict(self, X):
    """Make predictions for given input data."""
    a = X
    for w, b in zip(self.weights, self.biases):
      a = self.activation(np.dot(a, w) + b)
    return a


# --------------------- Example Usage ---------------------

if __name__ == "__main__":
  # Example: Learning the XOR function
  # XOR input and output
  X = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])
  y = np.array([[0],
                [1],
                [1],
                [0]])

  # Create a network with:
  #  - 2 input neurons (for 2 features)
  #  - 1 hidden layer with 2 neurons (sufficient for XOR with a non-linear activation)
  #  - 1 output neuron
  nn = NeuralNetwork(layer_sizes=[2, 2, 1], activation='tanh', learning_rate=0.1)

  # Train the network for 10,000 epochs
  nn.train(X, y, epochs=10000)

  # Make predictions
  predictions = nn.predict(X)
  print("\nFinal predictions:")
  print(predictions)
