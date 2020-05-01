import numpy as np
import matplotlib.pyplot as plt
import neural_network as nn
import scipy.io

## Set up parameters
input_layer_size = 400  # 20x20 Input Images of Digits
num_hidden_layers = 2   # 2 hidden layers
hidden_layer_size = 25  # 25 hidden units per layer
output_layer_size = 10  # 10 labels, from 0 to 9
lambda_reg = 1          # Weight regularization parameter
np.random.seed(1)
nn_layers_info = (input_layer_size, hidden_layer_size, num_hidden_layers, output_layer_size)


## Load training data
data = scipy.io.loadmat("ex4data1.mat")
X = data["X"]
y = data["y"]
y[y == 10] = 0

## Load theta weights into a list of thetas

## Load weights from file
# weights = scipy.io.loadmat("ex4weights.mat")
# thetas = tuple([weights["Theta1"], weights["Theta2"]])

# Random initialize theta weights from given neural network layers info
thetas = nn.create_hidden_layer_thetas(nn_layers_info)

## Unroll parameters
nn_params = nn.unroll_parameters(thetas)

## Training neural network
# optimal_thetas = nn.fmincg(nn_params, X, y, nn_layers_info, lambda_reg=1)  # Using advanced algorithm fmincg
optimal_thetas = nn.gradient_decsend(nn_params, X, y, nn_layers_info,       # Handwirtten gradient descend
                                        lambda_reg, 0.5, 500, plot=True)

## Predict - Compare result to training labels
prediction = nn.predict(optimal_thetas, X)
accuracy = np.mean(prediction == y) * 100
print(accuracy)

## Save trained weights
if accuracy >= 90:
    np.save("20x20_weights.npy", nn.unroll_parameters(optimal_thetas))