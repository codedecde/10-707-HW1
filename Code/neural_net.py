import numpy as np
import copy
from sklearn.metrics import accuracy_score
from utils import Progbar
import activations

class layer(object):
    def __init__(self, input_dim, output_dim, activation="sigmoid"):
        l_val = np.sqrt(6) / (np.sqrt(input_dim + output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.uniform(low=-l_val, high=l_val, size=(input_dim, output_dim))
        self.b = np.ones((1, output_dim))
        if not hasattr(activations, activation):
            print "No support currently for activation %s. Defaulting to linear " % (activation)
            self. activation = getattr(activations, 'linear')
        else:
            self.activation = getattr(activations, activation)

    def update_params(self):
        lr = 0.001
        self.W = self.W - (lr * self.grad_w)
        self.b = self.b - (lr * self.grad_b)

    def zero_grads(self):
        self.grad_w = 0.
        self.grad_b = 0.

    def forward(self, input_tensor):
        '''
            input_tensor : batch_size x input_dim
            Computes the forward pass, and stores the information required for the backward pass
            output_tensor : batch_size x output_dim
        '''
        self.input = input_tensor
        self.z = np.dot(input_tensor, self.W) + self.b  # batch_size x output_dim
        return self.activation(self.z)

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
        self.grad_w = np.dot(self.input.transpose(), activation_grad) / output_gradient.shape[0]
        self.grad_b = np.sum(activation_grad, axis=0) / output_gradient.shape[0]
        return np.dot(activation_grad, self.W.transpose())

    def __call__(self, input_tensor):
        return self.forward(input_tensor)


class neural_net(object):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.hidden_layer = layer(input_dim, hidden_dim, "relu")
        self.output_layer = layer(hidden_dim, output_dim, "softmax")

    def train_batch(self, X, y):
        # Forward Propagation
        hidden_rep = self.hidden_layer(X)
        output = self.output_layer(hidden_rep)
        loss = -1. * (y * np.log(output))
        loss = np.sum(np.sum(loss)) / y.shape[1]
        # Backward Propagation
        # error_grad = y * -1. / (output)  # batch x num_classes
        error_grad = (output - y)
        mid_grad = self.output_layer.backward(error_grad)  # batch x hidden_dim
        start_grad = self.hidden_layer.backward(mid_grad)  # batch x input_dim
        # Now update the weights
        self.hidden_layer.update_params()
        self.output_layer.update_params()
        return loss

    def predict(self, X):
        hidden_rep = self.hidden_layer(X)
        output = self.output_layer(hidden_rep)
        output = np.argmax(output, axis=-1)
        return output

    def fit(self, X, y, X_val, y_val):
        # Shuffle the training data
        index = np.arange(X.shape[0])
        np.random.shuffle(index)
        X = X[index]
        y = y[index]
        batch_size = 50
        n_epochs = 200
        y_val = np.argmax(y_val, axis=-1)
        bar = Progbar(n_epochs)
        for epoch in xrange(n_epochs):
            losses = []
            for ix in xrange(0, X.shape[0], batch_size):
                batch_x = X[ix: ix + batch_size]
                batch_y = y[ix: ix + batch_size]
                self.hidden_layer.zero_grads()
                self.output_layer.zero_grads()
                loss = self.train_batch(batch_x, batch_y)
                losses.append(loss)
                preds = self.predict(X_val)
            bar.update(epoch + 1, values=[("mean_training_loss", sum(losses) / len(losses)), ("val_acc", accuracy_score(y_val, preds))])


def get_x_y(data_array):
    x = data_array[:, :-1]
    y = data_array[:, -1].reshape(data_array.shape[0], 1)
    y = np.array(data_array[:, -1], dtype='int').ravel()
    num_classes = np.max(y) + 1
    categorical_y = np.zeros((x.shape[0], num_classes))
    categorical_y[np.arange(x.shape[0]), y] = 1
    return x, categorical_y


if __name__ == "__main__":
    train_file = "Data/digitstrain.txt"
    val_file = "Data/digitsvalid.txt"
    train_x, train_y = get_x_y(np.genfromtxt(train_file, delimiter=","))
    val_x, val_y = get_x_y(np.genfromtxt(val_file, delimiter=","))
    nn = neural_net(train_x.shape[1], 100, 10)
    nn.fit(train_x, train_y, val_x, val_y)
