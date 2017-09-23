import matplotlib
import numpy as np
import cPickle as cp
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb

######################################################
# LOAD THE HISTORY 
history_file = 'History/History_hidden_100_dropout_1.00_batch_32_l2_0.0000_lr_0.100_momentum_0.000_activation_sigmoid_epochs_500.pkl'
history = cp.load(open(history_file))
#######################################################

#######################################################
# Plot the Train and Val Error vs epochs
train_loss, = plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label="train_loss", color='r')
val_loss, = plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label="val_loss", color='b')
axes = plt.gca()
axes.set_ylim([0., 2.5])
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Error')
plt.title('Entropy Error vs Epochs')
plt.legend([train_loss, val_loss], ["Training Entropy Loss", "Validation Entropy Loss"])
plt.savefig('Plots/Entropy_loss.png')
plt.close()
########################################################

########################################################
# Plot the Train and Val Classification Error vs epochs
train_acc, = plt.plot(range(1, len(history['train_acc']) + 1), 1. - np.array(history['train_acc']), label="train_acc", color='r')
val_acc, = plt.plot(range(1, len(history['val_acc']) + 1), 1. - np.array(history['val_acc']), label="val_acc", color='b')
axes = plt.gca()
axes.set_ylim([0., .5])
plt.xlabel('Epochs')
plt.ylabel('Classification Error')
plt.title('Classification Error vs Epochs')
plt.legend([train_acc, val_acc], ["Training Classification Loss", "Validation Classification Loss"])
plt.savefig('Plots/Classification_loss.png')
plt.close()
########################################################

########################################################
# Plot the weights of the best model
model_file = 'Parameters/Model_hidden_100_dropout_1.00_batch_32_l2_0.0000_lr_0.100_momentum_0.000_activation_sigmoid_acc_0.9140_epoch_302'
params = cp.load(open(model_file))
weights = params['hidden_0']['W']  # 784 x 100
cols = int(np.sqrt(weights.shape[1]))
rows = (weights.shape[1] // cols) if (weights.shape[1] % cols == 0) else (weights.shape[1] // cols) + 1
fig, axes = plt.subplots(nrows=rows, ncols=cols)
fig.suptitle('Weights %d x %d' % (weights.shape))
for row_ix, row in enumerate(axes):
    for col_ix, ax in enumerate(row):
        index = (row_ix * col_ix) + col_ix
        if index >= weights.shape[1]:
            ax.imshow(np.zeros((28, 28)), cmap='gray')
        else:
            ax.imshow(weights[:, index].reshape((28, 28)), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
plt.savefig('Plots/Weights.png')
