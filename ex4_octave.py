import numpy as np
import matplotlib.pyplot as plt
import neural_network as nn
import scipy.io

# Set up parameters
input_layer_size = 400
num_hidden_layers = 2
hidden_layer_size = 25
output_layer_size = 10
lambda_reg = 1
np.random.seed(1)
nn_layers_info = (input_layer_size, hidden_layer_size, num_hidden_layers, output_layer_size)

data = scipy.io.loadmat("ex4data1.mat")
X = data["X"]
y = data["y"]
y[y == 10] = 0
# weights = scipy.io.loadmat("ex4weights.mat")
# thetas = tuple([weights["Theta1"], weights["Theta2"]])
thetas = nn.create_hidden_layer_thetas(nn_layers_info)

nn_params = nn.unroll_parameters(thetas)

optimal_thetas = nn.fmincg(nn_params, X, y, nn_layers_info, lambda_reg=1)
#optimal_thetas = nn.gradient_decsend(nn_params, X, y, nn_layers_info, lambda_reg, 0.5, 300, plot=True)
optimal_thetas = nn.reshape_parameters(optimal_thetas, nn_layers_info)
prediction = nn.predict(optimal_thetas, X)
accuracy = np.mean(prediction == y) * 100
print(accuracy)
