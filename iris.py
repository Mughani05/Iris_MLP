import numpy as np
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split

# !pip install ucimlrepo
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from ucimlrepo import fetch_ucirepo

def sigmoid(x: int) -> float:
  return 1 / (1 + np.exp(-x))

def sigmoid_back(x: int) -> float:
  fwd = sigmoid(x)
  return fwd * (1-fwd)

class MLP:
  """Multi-Layer Perceptron Class"""

  def __init__(self, sizes: list):
    self.sizes = sizes
    self.num_layers = len(sizes)

    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x)
                    for x, y in zip(sizes[:-1], sizes[1:])]

  def feedforward(self, a: np.ndarray) -> np.ndarray:
    activations = [a]
    zs = []

    for b, w in zip(self.biases, self.weights):
        zs.append(np.dot(w, a)+b)
        a = sigmoid(np.dot(w, a)+b)
        activations.append(a)

    return a, activations, zs

  def backpropagation(self, inputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    delta_b = [np.zeros(b.shape) for b in self.biases]
    delta_w = [np.zeros(w.shape) for w in self.weights]
    # feedforward
    activation = inputs
    res, activations, zs = self.feedforward(activation)
    # backward pass
    delta = (activations[-1] - labels) * sigmoid_back(zs[-1])
    delta_b[-1] = delta
    delta_w[-1] = np.dot(delta, activations[-2].transpose())

    for l in range(2, self.num_layers):
        z = zs[-l]
        sp = sigmoid_back(z)
        delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
        delta_b[-l] = delta
        delta_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (delta_w, delta_b)

# Load the Iris dataset
df = pd.read_csv("iris.data")

'''
DataFrame shape: (150, 5)
[sepal_length, sepal_width, petal_length, petal_width, class]
[5.1,         3.5,         1.4,          0.2,         Iris-setosa]
[4.9,         3.0,         1.4,          0.2,         Iris-setosa]
...
[5.9,         3.0,         5.1,          1.8,         Iris-virginica]
'''

# Preprocess the data
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Labels

'''
X shape: (150, 4)
[[5.1, 3.5, 1.4, 0.2],
 [4.9, 3.0, 1.4, 0.2],
 ...
 [5.9, 3.0, 5.1, 1.8]]

y shape: (150,)
['Iris-setosa', 'Iris-setosa', ..., 'Iris-virginica']
'''

# Encode the labels
label_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = np.array([label_mapping[label] for label in y])

'''
y shape: (150,)
[0, 0, ..., 2]
'''

'''ALT ENCODE IMPL
new_y = []  # Create an empty list to store the new values

# Loop through each label in the original y
for label in y:
    # Look up the numeric value for this label in the label_mapping dictionary
    numeric_value = label_mapping[label]

    # Add this numeric value to our new list
    new_y.append(numeric_value)

# Convert the list to a numpy array
y = np.array(new_y)'''

# Convert features to the required shape (4, 1) for each sample
X = np.array([x.reshape((4, 1)) for x in X])

'''
X shape: (150, 4, 1)
[[[5.1],   [[4.9],   ...   [[5.9],
  [3.5],    [3.0],          [3.0],
  [1.4],    [1.4],          [5.1],
  [0.2]],   [0.2]],         [1.8]]]
'''

# One-hot encode the labels
y_encoded = np.eye(3)[y] #np.eye creates a 2D array with 1s on the diagonal and 0s elsewhere
y_encoded = np.array([y.reshape((3, 1)) for y in y_encoded])

'''
y_encoded shape: (150, 3, 1)
[[[1],   [[1],   ...   [[0],
  [0],    [0],          [0],
  [0]],   [0]],         [1]]]
'''

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.33, random_state=42)

'''
X_train shape: (100, 4, 1)
y_train shape: (100, 3, 1)

X_test shape: (50, 4, 1)
y_test shape: (50, 3, 1)
'''

# Initialize the network
n = MLP([4, 5, 3])

'''
Network architecture:
Input layer:  4 neurons
Hidden layer: 5 neurons
Output layer: 3 neurons

Weights shapes:
w1: (5, 4)
w2: (3, 5)

Biases shapes:
b1: (5, 1)
b2: (3, 1)
'''

# Train the network
epochs = 500
learning_rate = 0.1

for _ in range(epochs):
    for x, y in zip(X_train, y_train):
        # x shape: (4, 1)
        # y shape: (3, 1)
        dw, db = n.backpropagation(x, y)
        for i in range(len(n.weights)):
            n.weights[i] -= learning_rate * dw[i]
            n.biases[i] -= learning_rate * db[i]

# Evaluate the network
correct = 0
total = len(X_test)

for x, y in zip(X_test, y_test):
    # x shape: (4, 1)
    # y shape: (3, 1)
    predicted, _, _ = n.feedforward(x)
    # predicted shape: (3, 1)
    predicted_class = np.argmax(predicted)
    actual_class = np.argmax(y)
    correct += int(predicted_class == actual_class)

accuracy = correct / total
print(f"Accuracy: {accuracy:.2f}")
