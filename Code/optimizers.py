import numpy as np


class optimizer(object):
    def __init__(self, params, loss_type, lr=0.001, l2_penalty=0., momentum=0.):
        self.params = params
        self.loss_type = loss_type
        self.lr = lr
        self.l2_penalty = l2_penalty
        self.momentum = momentum
        self.previous_grad = None

    def loss(self, y, output):
        if self.loss_type == 'categorical_cross_entropy':
            loss = -1. * (y * np.log(output))
            loss_grad = (output - y)
        elif self.loss_type == 'binary_cross_entropy':
            loss = -1. * ((y * np.log(output)) + ((1. - y) * (np.log(1. - output))))
            loss_grad = (output - y) / (output * (1. - output))
        elif self.loss_type == 'mean_squared_error':
            loss = -0.5 * (output - y) * (output - y)
            loss_grad = output * (output - y)
        else:
            print "Error %s not supported. Reverting to categorical_cross_entropy" % (self.loss_type)
            loss = -1. * (y * np.log(output))
            loss_grad = (output - y)
        loss = np.sum(np.sum(loss)) / y.shape[1]
        return loss, loss_grad

    def zero_grads(self):
        for name in self.params:
            for weight in self.params[name]:
                self.params[name][weight].grad = 0.

    def step(self):
        if self.previous_grad is None:
            self.previous_grad = {}
            for name in self.params:
                self.previous_grad[name] = {}
                for weight in self.params[name]:
                    if weight != 'b':
                        gradient = ((self.lr * self.params[name][weight].grad) + (self.l2_penalty * self.params[name][weight].value))
                        self.previous_grad[name][weight] = gradient
                        self.params[name][weight].value -= gradient
                    else:
                        gradient = (self.lr * self.params[name][weight].grad)
                        self.previous_grad[name][weight] = gradient
                        self.params[name][weight].value -= gradient
        else:
            for name in self.params:
                for weight in self.params[name]:
                    if weight != 'b':
                        gradient = ((self.lr * self.params[name][weight].grad) + (self.l2_penalty * self.params[name][weight].value))
                        gradient += self.momentum * self.previous_grad[name][weight]
                        self.params[name][weight].value -= gradient
                    else:
                        gradient = (self.lr * self.params[name][weight].grad)
                        gradient += self.momentum * self.previous_grad[name][weight]
                        self.params[name][weight].value -= gradient
