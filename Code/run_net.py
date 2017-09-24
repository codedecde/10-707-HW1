from neural_net import neural_net
import numpy as np
import argparse
import sys
import pdb
import cPickle as cp
# np.random.seed(1234)


def get_x_y(data_array):
    x = data_array[:, :-1]
    y = data_array[:, -1].reshape(data_array.shape[0], 1)
    y = np.array(data_array[:, -1], dtype='int').ravel()
    num_classes = np.max(y) + 1
    categorical_y = np.zeros((x.shape[0], num_classes))
    categorical_y[np.arange(x.shape[0]), y] = 1
    return x, categorical_y


def get_arguments():
    def check_boolean(args, attr_name):
        assert hasattr(args, attr_name), "%s not found in parser" % (attr_name)
        bool_set = set(["true", "false"])
        args_value = getattr(args, attr_name)
        args_value = args_value.lower()
        assert args_value in bool_set, "Boolean argument required for attribute %s" % (attr_name)
        args_value = False if args_value == "false" else True
        setattr(args, attr_name, args_value)
        return args

    parser = argparse.ArgumentParser(description='Basic Neural Network')
    parser.add_argument('-n_hidden', action="store", default=100, dest="n_hidden", type=int)
    parser.add_argument('-batch', action="store", default=32, dest="batch_size", type=int)
    parser.add_argument('-dropout', action="store", default=1.0, dest="dropout", type=float)
    parser.add_argument('-l2', action="store", default=0.000, dest="l2", type=float)
    parser.add_argument('-lr', action="store", default=0.1, dest="lr", type=float)
    parser.add_argument('-momentum', action="store", default=0.0, dest="momentum", type=float)
    parser.add_argument('-activation', action="store", default="sigmoid", dest="activation", type=str)
    parser.add_argument('-n_epochs', action="store", default=200, dest="n_epochs", type=int)
    parser.add_argument('-save_prefix', action="store", default="", dest="save_prefix", type=str)
    # Using strings as a proxy for boolean flags. Checks happen later
    args = parser.parse_args(sys.argv[1:])
    # Checks for the boolean flags
    return args


if __name__ == "__main__":
    opts = get_arguments()
    data_dir = "/home/bass/DataDir/10-707/HW1/"
    train_file = data_dir + "Data/digitstrain.txt"
    val_file = data_dir + "Data/digitsvalid.txt"
    train_x, train_y = get_x_y(np.genfromtxt(train_file, delimiter=","))
    val_x, val_y = get_x_y(np.genfromtxt(val_file, delimiter=","))
    layer_info = [("hidden", opts.n_hidden, opts.activation, opts.dropout), ("output", 10, "softmax", 1.)]
    # layer_info = [("hidden", 100, "relu", 0.5), ("batchnorm", 100, "", 0.), ("output", 10, "softmax", 1.)]
    # layer_info = [("hidden", 100, "tanh", .5), ("batchnorm", 100), ("hidden", 100, "tanh", .5), ("batchnorm", 100), ("output", 10, "softmax", 1.)]
    # layer_info = [("hidden", 5, "relu", 1.), ("batchnorm", 5), ("hidden", 5, "tanh", 1.), ("output", 10, "softmax", 1.)]
    # layer_info = [("hidden", 100, "relu", .5), ("batchnorm", 100), ("hidden", 100, "relu", .5), ("output", 10, "softmax", 1.)]
    # layer_info = [("hidden", 100, "relu", 1.), ("batchnorm", 100), ("hidden", 100, "relu", 1.), ("output", 10, "softmax", 1.)]
    # layer_info = [("hidden", 100, "relu", 1.), ("output", 10, "softmax", 1.)]
    model_save_prefix = data_dir + 'Parameters/Model_hidden_%d_dropout_%.2f_batch_%d_l2_%.4f_lr_%.3f_momentum_%.3f_activation_%s_' % (opts.n_hidden,
                                                                                                                                      opts.dropout,
                                                                                                                                      opts.batch_size,
                                                                                                                                      opts.l2,
                                                                                                                                      opts.lr,
                                                                                                                                      opts.momentum,
                                                                                                                                      opts.activation)
    opts.save_prefix = model_save_prefix if opts.save_prefix == "" else opts.save_prefix
    nn = neural_net(train_x.shape[1], layer_info, opts)
    history = nn.fit(train_x, train_y, val_x, val_y, n_epochs=opts.n_epochs, batch_size=opts.batch_size, return_history=True)
    history_file_name = 'History_hidden_%d_dropout_%.2f_batch_%d_l2_%.4f_lr_%.3f_momentum_%.3f_activation_%s_epochs_%d.pkl' % (opts.n_hidden,
                                                                                                                               opts.dropout,
                                                                                                                               opts.batch_size,
                                                                                                                               opts.l2,
                                                                                                                               opts.lr,
                                                                                                                               opts.momentum,
                                                                                                                               opts.activation,
                                                                                                                               opts.n_epochs)
    history_file_name = data_dir + 'History/' + history_file_name
    cp.dump(history, open(history_file_name, 'wb'))
