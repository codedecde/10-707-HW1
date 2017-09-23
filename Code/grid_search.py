from run_net import get_arguments, get_x_y
import neural_net as nn
import numpy as np
import random
import cPickle as cp


def build_strings(opts):
    model_save_prefix = data_dir + 'Parameters/Model_hidden_%d_dropout_%.2f_batch_%d_l2_%.4f_lr_%.3f_momentum_%.3f_activation_%s_' % (opts.n_hidden,
                                                                                                                                      opts.dropout,
                                                                                                                                      opts.batch_size,
                                                                                                                                      opts.l2,
                                                                                                                                      opts.lr,
                                                                                                                                      opts.momentum,
                                                                                                                                      opts.activation)
    history_file_name = data_dir + 'History/History_hidden_%d_dropout_%.2f_batch_%d_l2_%.4f_lr_%.3f_momentum_%.3f_activation_%s_epochs_%d.pkl' % (opts.n_hidden,
                                                                                                                                                  opts.dropout,
                                                                                                                                                  opts.batch_size,
                                                                                                                                                  opts.l2,
                                                                                                                                                  opts.lr,
                                                                                                                                                  opts.momentum,
                                                                                                                                                  opts.activation,
                                                                                                                                                  opts.n_epochs)
    return model_save_prefix, history_file_name


if __name__ == "__main__":
    data_dir = "/home/bass/DataDir/10-707/HW1/"
    train_file = data_dir + "Data/digitstrain.txt"
    val_file = data_dir + "Data/digitsvalid.txt"
    train_x, train_y = get_x_y(np.genfromtxt(train_file, delimiter=","))
    val_x, val_y = get_x_y(np.genfromtxt(val_file, delimiter=","))
    layer_units = [50, 100, 200, 500]
    lr_values = [0.1, 0.5, 0.01, 0.005]
    num_layers = 1
    l2_low = 0.0001
    l2_high = 0.01
    momentum_low = 0.6
    momentum_high = 0.95
    opts = get_arguments()
    models_per_unit = 1
    best_val_acc = None
    best_save_string = ''
    if num_layers == 1:
        for unit in layer_units:
            opts.n_hidden = unit
            layer_info = [('hidden', opts.n_hidden, opts.activation, 1.), ("output", 10, "softmax", 1.)]
            for _ in xrange(models_per_unit):
                opts.lr = random.choice(lr_values)
                opts.l2 = float(format(np.random.uniform(low=l2_low, high=l2_high), '.5f'))
                opts.momentum = float(format(np.random.uniform(low=momentum_low, high=momentum_high), '.3f'))
                print_string = "\nNUM HIDDEN UNITS %d" % (opts.n_hidden)
                print_string += "\nLR %.4f" % (opts.lr)
                print_string += "\nL2 %.5f" % (opts.l2)
                print_string += "\nMomentum %.3f" % (opts.momentum)
                print print_string
                model_save_prefix, history_file_name = build_strings(opts)
                opts.save_prefix = data_dir + 'GridSearch/' + model_save_prefix
                nnet = nn.neural_net(train_x.shape[1], layer_info, opts)
                history = nnet.fit(train_x, train_y, val_x, val_y, n_epochs=opts.n_epochs, batch_size=opts.batch_size, return_history=True)
                best_val_acc_model = history['best_val_acc']
                if best_val_acc is None or best_val_acc_model > best_val_acc:
                    best_val_acc = best_val_acc_model
                    best_save_string = model_save_prefix
                cp.dump(history, open(data_dir + 'GridSearch/' +  history_file_name, 'wb'))
    print '\n\n Best Validation Accuracy Achieved : %.4f\nBy Model File\n%s\n' % (best_val_acc, best_save_string)
