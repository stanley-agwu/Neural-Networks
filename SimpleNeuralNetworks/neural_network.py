import math
import random

class NeuralNetwork():
  def __init__(self):
    # Seed the random number generator, so we get the same random number each time
    random.seed(1)

    # Create 3 weights and set them to random values in the range -1 and +1
    self.weights = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

  # Make a prediction function
  def think(self, neuron_inputs):
    sum_of_weighted_inputs = self.__sum_of_weighted_inputs(neuron_inputs)
    neuron_output = self.__sigmoid(sum_of_weighted_inputs)
    return neuron_output

  def train(self, training_set_examples, number_of_iterations):
    for iteration in range(number_of_iterations):
      for training_set_example in training_set_examples:

        # Predict the output based on the training set example inputs
        predicted_output = self.think(training_set_example["inputs"])

        # Calculate the error as the difference between the desired output and the predicted output
        error_in_output = training_set_example["output"] - predicted_output

        # Iterate through the weights and adjust each one
        for index in range(len(self.weights)):

          # Get the neuron's input associated with this weight
          neuron_input = training_set_example["inputs"][index]

          # Calculate how much to adjust the weights by using the delta rule (gradient descent)
          adjustment_weight = neuron_input * error_in_output * self.__sigmoid_gradient(predicted_output)

          # Adjust the weight
          self.weights[index] += adjustment_weight


  # Calculate the sigmoid (Activation function)
  def __sigmoid(self, sum_of_weighted_inputs):
    return 1 / (1 + math.exp(-sum_of_weighted_inputs))

  # Calculate the gradient of the sigmoid using its own output
  def __sigmoid_gradient(self, neuron_output):
    return neuron_output * (1 - neuron_output)

  # Calculate sum of weighted inputs by multiplying each input with its own weight and sum the total
  def __sum_of_weighted_inputs(self, neuron_inputs):
    sum_of_weighted_input = 0
    for index, neuron_input in enumerate(neuron_inputs):
      sum_of_weighted_input += self.weights[index] * neuron_input
    return sum_of_weighted_input

neural_network = NeuralNetwork()

print("Random starting weights: " + str(neural_network.weights))

# The neural network training sets of 5 examples - to learn the pattern
training_set_examples = [
  {"inputs": [1, 1, 1], "output": 1},
  {"inputs": [0, 0, 1], "output": 0},
  {"inputs": [1, 0, 1], "output": 1},
  {"inputs": [0, 1, 1], "output": 0},
  {"inputs": [0, 0, 0], "output": 0},
  {"inputs": [1, 0, 1], "output": 1},
  {"inputs": [0, 1, 0], "output": 0},
]

# Train the neural network using 10, 000 iterations
neural_network.train(training_set_examples, number_of_iterations=100000)

print("New weights after training: " + str(neural_network.weights))

# Make predictions for new situations
new_situation_1 = [1, 1, 0]
new_situation_2 = [1, 0, 0]
new_situation_3 = [0, 0, 0]
new_situation_4 = [0, 1, 1]

prediction_1 = neural_network.think(new_situation_1)
print("Prediction for new situation 1 [1, 1, 0]: " + str(prediction_1))

prediction_2 = neural_network.think(new_situation_2)
print("Prediction for new situation 1 [1, 0, 0]: " + str(prediction_2))

prediction_3 = neural_network.think(new_situation_3)
print("Prediction for new situation 3 [0, 0, 0]: " + str(prediction_3))

prediction_4 = neural_network.think(new_situation_4)
print("Prediction for new situation 4 [0, 1, 1]: " + str(prediction_4))