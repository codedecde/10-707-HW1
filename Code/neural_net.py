import numpy as np
from sklearn.metrics import accuracy_score
from utils import Progbar
import pdb
import activations
from optimizers import optimizer


class variable(object):
    def __init__(self):
        self.value = None
        self.grad = None
    def __str__(self):
        return "Value : %s\nGradient: %s" % (str(self.value), str(self.grad))


class layer(object):
    def __init__(self, input_dim, output_dim, activation="sigmoid"):
        l_val = np.sqrt(6) / (np.sqrt(input_dim + output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {"W": variable(), "b": variable()}
        self.params["W"].value = np.random.uniform(low=-l_val, high=l_val, size=(input_dim, output_dim))
        self.params["b"].value = np.ones((1, output_dim))
        if not hasattr(activations, activation):
            print "No support currently for activation %s. Defaulting to linear " % (activation)
            self. activation = getattr(activations, 'linear')
        else:
            self.activation = getattr(activations, activation)

    def forward(self, input_tensor):
        '''
            input_tensor : batch_size x input_dim
            Computes the forward pass, and stores the information required for the backward pass
            output_tensor : batch_size x output_dim
        '''
        self.input = input_tensor
        self.z = np.dot(input_tensor, self.params["W"].value) + self.params["b"].value  # batch_size x output_dim
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
        self.params["W"].grad = np.dot(self.input.transpose(), activation_grad) / output_gradient.shape[0]
        self.params["b"].grad = np.sum(activation_grad, axis=0) / output_gradient.shape[0]
        return np.dot(activation_grad, self.params["W"].value.transpose())

    def __call__(self, input_tensor):
        return self.forward(input_tensor)


class neural_net(object):
    def __init__(self, input_dims, layers_info):
        self.layers_info = layers_info
        self.num_layers = len(layers_info)
        self.params = {}
        for ix in xrange(len(layers_info)):
            if ix == 0:
                input_dim = input_dims
            else:
                input_dim = layers_info[ix - 1][1]
            output_dim = layers_info[ix][1]
            layer_object = layer(input_dim, output_dim, layers_info[ix][2])
            self.params[layers_info[ix][0]] = layer_object.params
            setattr(self, 'layer_{}'.format(ix), layer_object)
        self.optimizer = optimizer(self.params, 'binary_cross_entropy', lr=0.001, l2_penalty=0)

    def forward_prop(self, input_tensor):
        output = input_tensor
        for ix in xrange(self.num_layers):
            output = getattr(self, 'layer_{}'.format(ix))(output)
        return output

    def backward_prop(self, loss_grad):
        back_grad = loss_grad
        for ix in xrange(self.num_layers - 1, -1, -1):
            back_grad = getattr(self, 'layer_{}'.format(ix)).backward(back_grad)

    def train_batch(self, X, y):
        self.optimizer.zero_grads()
        output = self.forward_prop(X)
        loss, loss_grad = self.optimizer.loss(y, output)
        self.backward_prop(loss_grad)
        self.optimizer.step()
        return loss

    def predict(self, X):
        output = self.forward_prop(X)
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
                # self.hidden_layer.zero_grads()
                # self.output_layer.zero_grads()
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
    layer_info = [("hidden", 100, "relu"), ("output", 10, "sigmoid")]
    nn = neural_net(train_x.shape[1], layer_info)
    nn.fit(train_x, train_y, val_x, val_y)
