'''
model_save_prefix = 'Parameters/Model_hidden_%d_dropout_%.2f_batch_%d_l2_%.4f_lr_%.3f_momentum_%.3f_activation_%s_' % (opts.n_hidden,
                                                                                                                           opts.dropout,
                                                                                                                           opts.batch_size,
                                                                                                                           opts.l2,
                                                                                                                           opts.lr,
                                                                                                                           opts.momentum,
                                                                                                                           opts.activation)
history_file_name = 'History/History_hidden_%d_dropout_%.2f_batch_%d_l2_%.4f_lr_%.3f_momentum_%.3f_activation_%s_epochs_%d.pkl' % (opts.n_hidden,
                                                                                                                                       opts.dropout,
                                                                                                                                       opts.batch_size,
                                                                                                                                       opts.l2,
                                                                                                                                       opts.lr,
                                                                                                                                       opts.momentum,
                                                                                                                                       opts.activation,
                                                                                                                                       opts.n_epochs)
model_file = self.save_prefix + "acc_%.4f_epoch_%d" % (val_acc, epoch + 1)

History_hidden1_50_hidden2_50_l2_0.0024_lr_0.100_momentum_0.918.pkl
'''
import os
import cPickle as cp
import numpy as np
import pdb
from run_net import get_arguments, get_x_y
from neural_net import neural_net
from sklearn.metrics import accuracy_score


class OneRun(object):
    def __init__(self, num_layers=1):
        self.num_layers = num_layers
        if self.num_layers == 1:
            self.hidden = None
        else:
            self.hidden1 = None
            self.hidden2 = None
        self.l2 = None
        self.lr = None
        self.momentum = None
        self.best_val_acc = None
        self.best_epoch = None
        self._train_acc = None
        self._test_acc = None
        self._val_loss = None
        self._train_loss = None
        self._test_loss = None
        self.activation = None
        self.model_file_name = None

    def set_params(self, filename, opts, dirname, test_x=None, test_y=None, test_labels_y=None):
        history = cp.load(open(dirname + filename))
        self.best_val_acc = history['best_val_acc']
        filename = filter(lambda x: x != "batchnorm", filename.rstrip('.pkl').split('_'))
        for ix in xrange(1, len(filename), 2):
            if hasattr(self, filename[ix]):
                if filename[ix] == "activation":
                    setattr(self, filename[ix], filename[ix + 1])
                else:
                    setattr(self, filename[ix], eval(filename[ix + 1]))
        history['val_acc'] = np.array(history['val_acc'])
        self.best_epoch = np.argmax(history['val_acc']) + 1
        assert np.max(history['val_acc']) == history['best_val_acc'], "Reported val acc {} Found {}".format(np.max(history['val_acc']), history['best_val_acc'])
        self._train_acc = history['train_acc'][self.best_epoch - 1]
        self._train_loss = history['train_loss'][self.best_epoch - 1]
        self._val_loss = history['val_loss'][self.best_epoch - 1]
        self.model_file_name = history['Model_Save_Prefix'] + "acc_%.4f_epoch_%d" % (self.best_val_acc, self.best_epoch)
        local_file_name = 'Parameters/' + self.model_file_name.split('/')[-1]
        opts.activation = self.activation if self.activation is not None else opts.activation
        if os.path.isfile(local_file_name):
            if self.num_layers == 1:
                opts.n_hidden = self.hidden
                layer_info = [("hidden", opts.n_hidden, opts.activation, opts.dropout), ("output", 10, "softmax", 1.)]
            else:
                opts.hidden1 = self.hidden1
                opts.hidden2 = self.hidden2
                layer_info = [('hidden', opts.hidden1, opts.activation, 1.), ('hidden', opts.hidden2, opts.activation, 1.), ("output", 10, "softmax", 1.)]
            opts.l2 = self.l2
            opts.lr = self.lr
            opts.momentum = self.momentum
            nn = neural_net(test_x.shape[1], layer_info, opts)
            nn.load_params(local_file_name)
            test_probs = nn.forward(test_x, test=True)
            self._test_loss, _ = nn.optimizer.loss(test_y, test_probs)
            preds = np.argmax(test_probs, axis=-1)
            self._test_acc = accuracy_score(test_labels_y, preds)


def write2csv(runs, csv_filename, model_file_name, num_layers):
    # f = open(csv_filename, 'wb')
    # g = open(model_file_name, 'wb')
    THRESHOLD = 5
    if num_layers == 1:
        csv_buf = "%s," % ("Hidden Units")
    else:
        csv_buf = "%s,%s," % ("H1", "H2")
    csv_buf += "%s,%s,%s," % ("Epoch", "L2", "LR")
    csv_buf += "%s,%s,%s," % ("Momentum", "Train Loss", "Train Acc")
    csv_buf += "%s,%s," % ("Val Loss", "Val Acc")
    if any(x._test_acc is None for x in runs[:THRESHOLD]):
        csv_buf += "\n"
    else:
        csv_buf += "%s,%s\n" % ("Test Loss", "Test Acc")
    # f.write(csv_buf)
    for ix in xrange(len(runs)):
        if ix == THRESHOLD:
            break
        run = runs[ix]
        # Write to terminal
        if num_layers == 1:
            write_buf = "\nNum Units   : %d" % run.hidden
        else:
            write_buf = "\nNum Units L1   : %d" % run.hidden1
            write_buf += "\nNum Units L2   : %d" % run.hidden2
        write_buf += "\nActivation : %s" % run.activation
        write_buf += "\nEpoch      : %d" % run.best_epoch
        write_buf += "\nL2         : %.4f" % run.l2
        write_buf += "\nLR         : %.4f" % run.lr
        write_buf += "\nMomentum   : %.4f" % run.momentum
        write_buf += "\nTrain Loss : %.4f" % run._train_loss
        write_buf += "\nVal Loss   : %.4f" % run._val_loss
        write_buf += "\nTrain Acc  : %.4f" % run._train_acc
        write_buf += "\nVal Acc    : %.4f" % run.best_val_acc
        if run._test_loss is not None:
            write_buf += "\nTest Loss  : %.4f" % run._test_loss
            write_buf += "\nTest Acc   : %.4f" % run._test_acc
        write_buf += "\nModel File Name\n %s" % run.model_file_name
        print write_buf
        # Write model to model file
        # g.write("%s\n" % (run.model_file_name))
        # Write to CSV
        if num_layers == 1:
            csv_buf = "%d" % (run.hidden)
        else:
            csv_buf = "%d,%d" % (run.hidden1, run.hidden2)
        csv_buf += ",%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (run.best_epoch, run.l2, run.lr, run.momentum, run._train_loss, run._train_acc, run._val_loss, run.best_val_acc)
        csv_buf += ",%.4f,%.4f\n" % (run._test_loss, run._test_acc) if run._test_loss is not None else "\n"
        # f.write(csv_buf)
    # f.close()
    # g.close()


if __name__ == "__main__":
    opts = get_arguments()
    test_file = "Data/digitstest.txt"
    test_x, test_y = get_x_y(np.genfromtxt(test_file, delimiter=","))
    test_y_labels = np.argmax(test_y, axis=-1)
    runs = []
    num_layers = 2
    csv_filename = "SingleRun.csv" if num_layers == 1 else "2LayerRun.csv"
    ModelFile = "RunSingleModels.txt" if num_layers == 1 else "Run2LayersModels.txt"
    dirname = 'History_Activations' if num_layers == 1 else "History_Activations_L2"
    if not dirname.endswith('/'):
        dirname += '/'
    for file in os.listdir(dirname):
        run = OneRun(num_layers)
        run.set_params(file, opts, dirname, test_x, test_y, test_y_labels)
        runs.append(run)
    runs = sorted(runs, key=lambda x: x.best_val_acc, reverse=True)
    write2csv(runs, csv_filename, ModelFile, num_layers)
