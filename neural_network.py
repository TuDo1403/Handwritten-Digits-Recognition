import numpy as np
from copy import deepcopy
import scipy.optimize as op
import matplotlib.pyplot as plt
    
def rand_initial_weights(layer_in, layer_out, epsilon_init=0.12):
    """ Randomly initialize the weights of a layer with L_in incoming connections and L_out outgoing connections

        Parameters:
        epsilon_init (float): Range where random values varies

        Returns:
        Theta (ndarray): A ndarray theta with random values in range epsilon

    """
    
    return np.random.rand(layer_out, layer_in+1) * 2*epsilon_init - epsilon_init

def sigmoid(z):
    """ Compute sigmoid function of z

        Parameters: 
        z (ndarray): z where z = X * Theta

        Returns: 
        g(z) (ndarray): Returning values

    """

    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    """ Compute derivative of g(z) - sigmoid of z

        Parameters:
        z (ndarray): z where z = X * Theta

        Returns:
        g(z)' (ndarray): Returning values
        
    """

    return sigmoid(z) * (1 - sigmoid(z))

def unroll_parameters(thetas):
    """ Unroll list of ndarray thetas into 1d array

        Parameters:
        thetas (list/tupple): list of thetas weights

        Returns:
        nn_params (1d array): 1d array of thetas

    """
    nn_params = np.array([])
    for theta in thetas:
        nn_params = np.append(nn_params, theta.flatten())
    return nn_params

def create_hidden_layer_thetas(nn_layers_info):
    input_layer_size = nn_layers_info[0]
    hidden_layer_size = nn_layers_info[1]
    num_hidden_layers = nn_layers_info[2]
    output_layer_size = nn_layers_info[3]

    thetas = []
    if num_hidden_layers == 1:
        return rand_initial_weights(input_layer_size, output_layer_size)

    for i in range(num_hidden_layers):
        if i == 0:
            thetas.append(rand_initial_weights(input_layer_size, hidden_layer_size))
        elif i == num_hidden_layers-1:
            thetas.append(rand_initial_weights(hidden_layer_size, output_layer_size))
        else:
            thetas.append(rand_initial_weights(hidden_layer_size, hidden_layer_size))
    return tuple(thetas)

def reshape_parameters(nn_params, nn_layers_info):
    input_layer = nn_layers_info[0]
    hidden_layer = nn_layers_info[1]
    num_hidden_layers = nn_layers_info[2]
    output_layer = nn_layers_info[3]

    thetas = []
    if num_hidden_layers == 1:
        return np.reshape(thetas, (output_layer, input_layer+1))

    slice_idx = 0
    for i in range(num_hidden_layers):
        if i == 0:
            thetas.append(np.reshape(nn_params[: hidden_layer * (input_layer+1)], (hidden_layer, input_layer+1)))
            slice_idx += hidden_layer * (input_layer+1)
        elif i == num_hidden_layers-1:
            temp = nn_params[slice_idx:]
            thetas.append(np.reshape(nn_params[slice_idx :], (output_layer, hidden_layer+1)))
        else:
            thetas.append(np.reshape(nn_params[slice_idx : slice_idx + hidden_layer * (hidden_layer+1)], (hidden_layer, hidden_layer+1)))
            slice_idx += hidden_layer * (hidden_layer+1)
    return tuple(thetas)

def create_gradient_thetas(thetas):
    theta_grad = list(map(lambda theta : np.zeros(np.shape(theta)), thetas))
    return theta_grad

def forward_propagation(X, thetas):
    layers_activation = []

    # init activation value for input units (a0)
    a = np.hstack((np.ones((len(X), 1)), X))
    layers_activation.append(a)
    
    for idx, theta in enumerate(thetas):
        z = a.dot(theta.T)
        a = sigmoid(z)
        if idx != len(thetas)-1:
            a = np.hstack((np.ones((len(a), 1)), a))
        layers_activation.append(a)

    return layers_activation

def back_propagation(i, y, thetas, thetas_grad, a):
    new_thetas_grad = list(deepcopy(thetas_grad))
    deltas = []

    for j in range(-1, -len(thetas_grad)-1, -1):
        delta = 0
        a_j = a[j-1][i, np.newaxis]
        if j == -1:
            delta = a[j][i, np.newaxis] - y[i, np.newaxis]
            new_thetas_grad[j] += delta.T.dot(a_j)
        else:
            delta = deltas[-1].dot(thetas[j+1]) * a[j][i, np.newaxis] * (1-a[j][i, np.newaxis])
            delta = delta[:, 1:] # remove/skip delta 0
            new_thetas_grad[j] += (delta.T).dot(a_j)
        deltas.append(delta)
        
    return new_thetas_grad

def create_training_sets(y, num_labels):
    m = len(y)
    y_train = np.zeros((m, num_labels))
    for i in range(m):
        y_train[i, y[i]] = 1

    return y_train


def nn_cost_function(nn_params, X, y, nn_layers_info, lambda_reg=0):
    y_train = create_training_sets(y, nn_layers_info[-1])
    thetas = reshape_parameters(nn_params, nn_layers_info) 
    
    h = forward_propagation(X, thetas)[-1]  # list of activation units each layer
    m, J = len(X), 0
    for i in range(m):
        J += -1/m * (y_train[i].dot(np.log(h[i]).T) + (1-y_train[i]).dot(np.log(1-h[i]).T))

    temps = deepcopy(thetas)
    for temp in temps:
        temp[:, 0] = 0

    unrolled_temps = unroll_parameters(temps)
    J += (0.5*lambda_reg)/m * sum(unrolled_temps **2)

    return J

def nn_gradient(nn_params, X, y, nn_layers_info, lambda_reg=0):
    thetas = reshape_parameters(nn_params, nn_layers_info)
    thetas_grad = create_gradient_thetas(thetas)
    y_train = create_training_sets(y, nn_layers_info[-1])

    a = forward_propagation(X, thetas)
    m = len(X)
    for i in range(m):
        thetas_grad = back_propagation(i, y_train, thetas, thetas_grad, a)

    temps = deepcopy(thetas)
    for temp in temps:
        temp[:, 0] = 0
    for i in range(len(thetas_grad)):
        thetas_grad[i] = 1/m * (thetas_grad[i] + lambda_reg * temps[i])

    return unroll_parameters(thetas_grad)

def fmincg(nn_params, X, y, nn_layers_info, lambda_reg):
    result = op.fmin_cg(f=nn_cost_function, x0=nn_params, maxiter=50, args=(X, y, nn_layers_info, lambda_reg), fprime=nn_gradient)
    optimal_theta = reshape_parameters(result, nn_layers_info)
    return optimal_theta

def predict(thetas, X):
    h = forward_propagation(X, thetas)[-1]
    result = np.argmax(h, axis=1)[:, np.newaxis]
    return result

def gradient_decsend(nn_params, X, y, nn_layers_info, lambda_reg, alpha, iterations, plot=False):
    J_history = np.zeros((iterations, 1))
    optimal_thetas = deepcopy(nn_params)
    if plot:
        plt.xlim([0, iterations])
        plt.xlabel("Iteration")
        plt.ylabel("$J(\Theta)$")
        plt.title("Cost function using Gradient Descent")

    for i in range(iterations):
        optimal_thetas -= alpha * nn_gradient(optimal_thetas, X, y, nn_layers_info, lambda_reg)
        J_history[i] = nn_cost_function(optimal_thetas, X, y, nn_layers_info, lambda_reg)
        if plot:
            plt.plot(i, J_history[i], 'b.')
            plt.pause(0.0001)
    if plot:        
        plt.show()
    return reshape_parameters(optimal_thetas, nn_layers_info)


    
        
    


    
