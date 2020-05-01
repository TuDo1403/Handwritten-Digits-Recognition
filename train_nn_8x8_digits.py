import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import neural_network as nn

digits = load_digits(n_class=10)

## Set up parameters
input_layer_size = digits.data.shape[1] # 8x8 Input Images of Digits
num_hidden_layers = 2                   # 2 hidden layers
hidden_layer_size = 25                  # 25 hidden units per layer
output_layer_size = 10                  # 10 labels, from 0 to 9
lambda_reg = 1                          # Weight regularization parameter
np.random.seed(1)
nn_layers_info = (input_layer_size, hidden_layer_size, num_hidden_layers, output_layer_size)

## Load training data
X = digits.data
y = digits.target[:, np.newaxis]

## Load theta weights into a list of thetas
thetas = nn.create_hidden_layer_thetas(nn_layers_info)

## Unroll parameters
nn_params = nn.unroll_parameters(thetas)

## Training neural network
# optimal_thetas = nn.fmincg(nn_params, X, y, nn_layers_info, lambda_reg=0)                             # Using advanced algorithm fmincg
optimal_thetas = nn.gradient_decsend(nn_params, X, y, nn_layers_info, lambda_reg, 0.53, 300, plot=True) # Handwritten gradient descend

## Predict - Compare result to training labels
prediction = nn.predict(optimal_thetas, X)
accuracy = np.mean(prediction == y) * 100
print("# Training set accuracy: {}".format(accuracy))

## Save trained weights
if accuracy >= 90:
    np.save("8x8_weights.npy", nn.unroll_parameters(optimal_thetas))


