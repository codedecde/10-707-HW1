import numpy as np
from sklearn.metrics import accuracy_score
from utils import Progbar
from optimizers import SGD
from Layers import DenseLayer, BatchNormLayer
from copy import deepcopy
import pdb


class neural_net(object):
    def __init__(self, input_dims, layers_info, opts):
        self.layers_info = layers_info
        self.num_layers = len(layers_info)
        self.params = {}
        for ix in xrange(len(layers_info)):
            if ix == 0:
                input_dim = input_dims
            else:
                input_dim = layers_info[ix - 1][1]
            output_dim = layers_info[ix][1]
            if layers_info[ix][0] != "batchnorm":
                layer_object = DenseLayer(input_dim, output_dim, layers_info[ix][2], dropout=layers_info[ix][3])
            else:
                layer_object = BatchNormLayer(input_dim)
            self.params[layers_info[ix][0] + "_{}".format(ix)] = layer_object.params
            setattr(self, 'layer_{}'.format(ix), layer_object)
        self.optimizer = SGD(self.params, 'categorical_cross_entropy', lr=opts.lr, l2_penalty=opts.l2)

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

    def test_layer_gradient(self, layer, param, X, y):
        max_abs_difference = -1
        for i in xrange(self.params[layer][param].value.shape[0]):
            for j in xrange(self.params[layer][param].value.shape[1]):
                num_gradient = self.compute_numerical_grad(layer, param, i, j, X, y)
                abs_difference = abs(num_gradient - self.params[layer][param].grad[i][j]) / abs(num_gradient + self.params[layer][param].grad[i][j] + np.finfo(float).eps)
                max_abs_difference = max(abs_difference, max_abs_difference)
        return max_abs_difference

    def train_batch(self, X, y, X_val, y_val):
        self.optimizer.zero_grads()
        output = self.forward(X)
        loss, loss_grad = self.optimizer.loss(y, output)
        self.backward(loss_grad)
        # print self.test_layer_gradient('hidden_0', 'b', X, y)
        self.optimizer.step()
        loss_val, _ = self.optimizer.loss(y_val, self.forward(X_val))
        return loss, loss_val

    def predict(self, X):
        output = self.forward(X, test=True)
        output = np.argmax(output, axis=-1)
        return output

    def fit(self, X_train, y_train, X_val, y_val, n_epochs=200, batch_size=32, return_history=False):
        y_labels_val = np.argmax(y_val, axis=-1)
        y_labels_train = np.argmax(y_train, axis=-1)
        bar = Progbar(n_epochs)
        if return_history:
            history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        for epoch in xrange(n_epochs):
            # Shuffle the training data
            index = np.arange(X_train.shape[0])
            np.random.shuffle(index)
            X = X_train[index]
            y = y_train[index]
            train_loss = 0.
            val_loss = 0.
            for ix in xrange(0, X.shape[0], batch_size):
                batch_x = X[ix: ix + batch_size]
                batch_y = y[ix: ix + batch_size]
                loss_train, loss_val = self.train_batch(batch_x, batch_y, X_val, y_val)
                train_loss += loss_train * batch_x.shape[0]
                val_loss += loss_val * batch_x.shape[0]
            train_loss /= X.shape[0]
            val_loss /= X.shape[0]
            train_acc = accuracy_score(y_labels_train, self.predict(X_train))
            val_acc = accuracy_score(y_labels_val, self.predict(X_val))
            if return_history:
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
            bar.update(epoch + 1, values=[("train_loss", train_loss),
                                          ("val_loss", val_loss),
                                          ("train_acc", train_acc),
                                          ("val_acc", val_acc)])
        if return_history:
            return history
