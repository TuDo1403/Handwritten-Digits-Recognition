import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import scipy.io
import neural_network as nn

digits = load_digits(n_class=10)

# Set up parameters
input_layer_size = digits.data.shape[1]
num_hidden_layers = 2
hidden_layer_size = 25
output_layer_size = 10
lambda_reg = 1
np.random.seed(1)
nn_layers_info = (input_layer_size, hidden_layer_size, num_hidden_layers, output_layer_size)

X = digits.data
y = digits.target[:, np.newaxis]

thetas = nn.create_hidden_layer_thetas(nn_layers_info)
nn_params = nn.unroll_parameters(thetas)
#optimal_thetas = nn.fmincg(nn_params, X, y, nn_layers_info, lambda_reg=0)
optimal_thetas = nn.gradient_decsend(nn_params, X, y, nn_layers_info, 2, 0.53, 500, plot=True)
optimal_thetas = nn.reshape_parameters(optimal_thetas, nn_layers_info)


# print(nn.sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1])))
# grad = nn.nn_gradient(nn_params, X, y, nn_layers_info, lambda_reg=1)
# print(grad)

#nn_params = nn.gradient_decsend(nn_params, X, y, nn_layers_info, 2, 0.53, 300, plot=True)
# nn_params = nn.fmincg(nn_params, X, y, nn_layers_info, lambda_reg=0)
#optimal_thetas = nn.reshape_parameters(nn_params, nn_layers_info)

prediction = nn.predict(optimal_thetas, X)
accuracy = np.mean(prediction == y) * 100
print("# Training set accuracy: {}".format(accuracy))

