import numpy as np
import activations


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
        self.dropout = dropout
        if not hasattr(activations, activation):
            print "No support currently for activation %s. Defaulting to linear " % (activation)
            self. activation = getattr(activations, 'linear')
        else:
            self.activation = getattr(activations, activation)

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
        if self.activation.__name__ == "sigmoid":
            activation_grad = output_gradient * self.activation(self.z) * (1. - self.activation(self.z))
        elif self.activation.__name__ == "relu":
            activation_grad = output_gradient
            activation_grad[self.z < 0.] = 0.
        elif self.activation.__name__ == "tanh":
            activation_grad = output_gradient * (1. - (self.activation(self.z)**2))
        elif self.activation.__name__ == "softmax":
            activation_grad = output_gradient
        elif self.activation.__name__ == "linear":
            activation_grad = output_gradient
        else:
            print "Warning %s activation not found. Defaulting to linear " % (self.activation.__name__)
            activation_grad = output_gradient
        # Now compute the gradients of w and b and store those
        self.params["W"].grad = np.dot(self.input.transpose(), activation_grad) / output_gradient.shape[0]
        self.params["b"].grad = np.sum(activation_grad, axis=0) / output_gradient.shape[0]
        return np.dot(activation_grad, self.params["W"].value.transpose())

    def __call__(self, input_tensor, test=False):
        return self.forward(input_tensor, test)


class BatchNormLayer(object):
    def __init__(self, n_dim, epsilon=0.001):
        self.n_dim = n_dim
        self.params = {"gamma": Variable(), "beta": Variable()}
        self.params["gamma"].value = np.ones((1, n_dim))
        self.params["beta"].value = np.ones((1, n_dim))
        self.buffers = {}
        self.buffers["mu"] = np.ones((1, n_dim))
        self.buffers["sigma"] = np.ones((1, n_dim))
        self.sum_of_squares = np.ones((1, n_dim))
        self.count = 0.
        self.epsilon = epsilon

    def forward(self, input_tensor, test=False):
        '''
            input_tensor : batch x n_dim
            Returns a tensor of form batch x n_dim
        '''
        if not test:
            batch_mu = np.sum(input_tensor, axis=0)  # 1 x n_dim
            batch_sum_of_squares = np.sum(input_tensor * input_tensor, axis=0)
            self.buffers["mu"] = ((self.buffers["mu"] * self.count) + batch_mu) / (self.count + input_tensor.shape[0])
            self.sum_of_squares = ((self.sum_of_squares * self.count) + batch_sum_of_squares) / (self.count + input_tensor.shape[0])
            self.count += input_tensor.shape[0]
            self.buffers["sigma"] = self.sum_of_squares - (self.buffers["mu"] * self.buffers["mu"])
        self.numerator = (input_tensor - self.buffers["mu"])
        self.denominator = self.buffers["sigma"] + self.epsilon
        self.input_tensor_hat = self.numerator / (np.sqrt(self.denominator))
        return (self.input_tensor_hat * self.params["gamma"].value) + self.params["beta"].value

    def backward(self, output_gradient):
        '''
            output_gradient : batch x n_dim
            Returns a tensor of form batch x n_dim, computing the gradient wrt current function
            Also updates gradients for gamma and mu
        '''
        self.params["beta"].grad = np.sum(output_gradient, axis=0) / output_gradient.shape[0]
        self.params["gamma"].grad = np.sum(self.input_tensor_hat * output_gradient, axis=0) / output_gradient.shape[0]
        retval = (1. / output_gradient.shape[0]) * self.params["gamma"].value / (np.sqrt(self.denominator))
        retval = ((output_gradient.shape[0] * output_gradient) - np.sum(output_gradient, axis=0) - (self.numerator / self.denominator * np.sum(output_gradient * self.numerator, axis=0))) * retval
        return retval

    def __call__(self, input_tensor, test=False):
        return self.forward(input_tensor, test)
