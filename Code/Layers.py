import numpy as np
import activations
import pdb


class Variable(object):
    def __init__(self):
        self.value = None
        self.grad = None

    def __str__(self):
        return "Value : %s\nGradient: %s" % (str(self.value), str(self.grad))


class DenseLayer(object):
    def __init__(self, input_dim, output_dim, activation="sigmoid", dropout=1.0):
        l_val = np.sqrt(6) / (np.sqrt(input_dim + output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {"W": Variable(), "b": Variable()}
        self.params["W"].value = np.random.uniform(low=-l_val, high=l_val, size=(input_dim, output_dim))
        self.params["b"].value = np.ones((1, output_dim))
        self.dropout = min(1., max(dropout, 0.))
        if not hasattr(activations, activation):
            print "No support currently for activation %s. Defaulting to linear " % (activation)
            self. activation = getattr(activations, 'linear')
        else:
            self.activation = getattr(activations, activation)
        if not hasattr(activations, activation + '_grad'):
            print "Warning %s activation not found. Defaulting to linear " % (self.activation.__name__)
            self.gradient_function = getattr(activations, 'linear_grad')
        else:
            self.gradient_function = getattr(activations, activation + '_grad')

    def forward(self, input_tensor, test=False):
        '''
            input_tensor : batch_size x input_dim
            Computes the forward pass, and stores the information required for the backward pass
            output_tensor : batch_size x output_dim
        '''
        self.input = input_tensor
        self.z = np.dot(input_tensor, self.params["W"].value) + self.params["b"].value  # batch_size x output_dim
        a = self.activation(self.z)
        if test:
            a = a * self.dropout
        else:
            mask = np.random.binomial(1, self.dropout, a.shape)
            a *= mask
        return a

    def backward(self, output_gradient):
        '''
            output_gradient : batch_size x output_dim
            Computes the backward pass and stores the gradients
            returns the gradient w.r.t current node
        '''
        # Step 1. Compute gradient wrt the activation
        activation_grad = output_gradient * self.gradient_function(self.z)
        # Now compute the gradients of w and b and store those
        self.params["W"].grad = np.dot(self.input.transpose(), activation_grad) / output_gradient.shape[0]
        self.params["b"].grad = (np.sum(activation_grad, axis=0) / output_gradient.shape[0]).reshape(self.params["b"].value.shape)
        return np.dot(activation_grad, self.params["W"].value.transpose())

    def __call__(self, input_tensor, test=False):
        return self.forward(input_tensor, test)


class BatchNormLayer(object):
    def __init__(self, n_dim, epsilon=0.001, alpha=0.9):
        self.n_dim = n_dim
        l_val = np.sqrt(6) / (np.sqrt(n_dim + n_dim))
        self.params = {"gamma": Variable(), "beta": Variable()}
        self.params["gamma"].value = np.random.uniform(low=-l_val, high=l_val, size=(1, n_dim))
        self.params["beta"].value = np.random.uniform(low=-l_val, high=l_val, size=(1, n_dim))
        self.buffers = {}
        self.sum_of_squares = np.zeros((1, n_dim))
        self.count = 0.
        self.alpha = alpha
        self.buffers["mu"] = np.zeros((1, n_dim))
        self.buffers["var"] = np.zeros((1, n_dim))
        self.epsilon = epsilon

    def forward(self, input_tensor, test=False):
        '''
            input_tensor : batch x n_dim
            Returns a tensor of form batch x n_dim
        '''
        if not test:
            batch_mu = np.sum(input_tensor, axis=0) / input_tensor.shape[0]
            self.buffers["mu"] = self.alpha * self.buffers["mu"] + ((1. - self.alpha) * batch_mu) if self.buffers["mu"] is not None else batch_mu
            batch_var = (np.sum(input_tensor ** 2, axis=0) / input_tensor.shape[0]) - (batch_mu ** 2)
            self.buffers["var"] = self.alpha * self.buffers["var"] + ((1. - self.alpha) * batch_var) if self.buffers["var"] is not None else batch_var
        else:
            batch_mu = self.buffers["mu"]
            batch_var = self.buffers["var"]
        self.x_minus_mu = input_tensor - batch_mu
        self.var_plus_eps = batch_var + self.epsilon
        self.x_hat = self.x_minus_mu / np.sqrt(self.var_plus_eps)
        output_tensor = (self.x_hat * self.params["gamma"].value) + self.params["beta"].value
        return output_tensor

    def backward(self, output_gradient):
        '''
            output_gradient : batch x n_dim
            Returns a tensor of form batch x n_dim, computing the gradient wrt current function
            Also updates gradients for gamma and mu
        '''
        self.params["beta"].grad = (np.sum(output_gradient, axis=0) / output_gradient.shape[0]).reshape(self.params["beta"].value.shape)
        self.params["gamma"].grad = (np.sum(self.x_hat * output_gradient, axis=0) / output_gradient.shape[0]).reshape(self.params["beta"].value.shape)
        grad_input_tensor_hat = output_gradient * self.params["gamma"].value  # batch x n_dim
        grad_var = -1. * np.sum(grad_input_tensor_hat * (self.x_minus_mu / (2 * (self.var_plus_eps * np.sqrt(self.var_plus_eps)))), axis=0)
        grad_mu = np.sum(grad_input_tensor_hat * (-1. / (np.sqrt(self.var_plus_eps))), axis=0)
        grad_mu += ((grad_var / output_gradient.shape[0]) * -2. * np.sum(self.x_minus_mu, axis=0))
        grad_input = (grad_input_tensor_hat / (np.sqrt(self.var_plus_eps)))
        grad_input += (grad_var * (2. / output_gradient.shape[0]) * self.x_minus_mu)
        grad_input += (grad_mu / output_gradient.shape[0])
        return grad_input

    def __call__(self, input_tensor, test=False):
        return self.forward(input_tensor, test)
