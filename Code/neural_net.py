import numpy as np
from sklearn.metrics import accuracy_score
from utils import Progbar
from optimizers import optimizer
from Layers import DenseLayer, BatchNormLayer
np.random.seed(1234)


DEBUG = False


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
            if layer_info[ix][0] != "batchnorm":
                layer_object = DenseLayer(input_dim, output_dim, layers_info[ix][2], dropout=layer_info[ix][3])
            else:
                layer_object = BatchNormLayer(input_dim)
            self.params[layers_info[ix][0] + "_{}".format(ix)] = layer_object.params
            setattr(self, 'layer_{}'.format(ix), layer_object)
        self.optimizer = optimizer(self.params, 'categorical_cross_entropy', lr=0.1, l2_penalty=0)

    def forward(self, input_tensor, test=False):
        output = input_tensor
        for ix in xrange(self.num_layers):
            output = getattr(self, 'layer_{}'.format(ix))(output, test=test)
        return output

    def backward(self, loss_grad):
        back_grad = loss_grad
        for ix in xrange(self.num_layers - 1, -1, -1):
            back_grad = getattr(self, 'layer_{}'.format(ix)).backward(back_grad)

    def compute_numerical_grad(self, layer, param, i, j, X, y, eps=0.0001):
        original_params = deepcopy(self.params[layer][param].value)
        self.params[layer][param].value[i][j] = original_params[i][j] + eps
        loss_pos, _ = self.optimizer.loss(y, self.forward(X))
        self.params[layer][param].value[i][j] = original_params[i][j] - eps
        loss_neg, _ = self.optimizer.loss(y, self.forward(X))
        num_grad = (loss_pos - loss_neg) / (2 * eps)
        self.params[layer][param].value = original_params
        return num_grad

    def train_batch(self, X, y):
        self.optimizer.zero_grads()
        output = self.forward(X)
        loss, loss_grad = self.optimizer.loss(y, output)
        self.backward(loss_grad)
        self.optimizer.step()
        return loss

    def predict(self, X):
        output = self.forward(X, test=True)
        output = np.argmax(output, axis=-1)
        return output

    def fit(self, X, y, X_val, y_val, n_epochs=200, batch_size=32):
        y_val = np.argmax(y_val, axis=-1)
        bar = Progbar(n_epochs)
        for epoch in xrange(n_epochs):
            # Shuffle the training data
            index = np.arange(X.shape[0])
            np.random.shuffle(index)
            X = X[index]
            y = y[index]
            losses = []
            for ix in xrange(0, X.shape[0], batch_size):
                batch_x = X[ix: ix + batch_size]
                batch_y = y[ix: ix + batch_size]
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
    # layer_info = [("hidden", 100, "relu", 0.5), ("batchnorm", 100, "", 0.), ("output", 10, "softmax", 1.)]
    # layer_info = [("hidden", 100, "tanh", .5), ("batchnorm", 100), ("hidden", 100, "tanh", .5), ("batchnorm", 100), ("output", 10, "softmax", 1.)]
    # layer_info = [("hidden", 5, "relu", 1.), ("batchnorm", 5), ("hidden", 5, "tanh", 1.), ("output", 10, "softmax", 1.)]
    # layer_info = [("hidden", 100, "relu", .5), ("batchnorm", 100), ("hidden", 100, "relu", .5), ("output", 10, "softmax", 1.)]
    # layer_info = [("hidden", 100, "relu", .5), ("hidden", 100, "relu", .5), ("output", 10, "softmax", 1.)]
    layer_info = [("hidden", 100, "relu", 1.), ("output", 10, "softmax", 1.)]
    nn = neural_net(train_x.shape[1], layer_info)
    nn.fit(train_x, train_y, val_x, val_y)
