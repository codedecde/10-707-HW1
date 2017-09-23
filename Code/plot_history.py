import matplotlib
import numpy as np
import cPickle as cp
matplotlib.use('Agg')
import matplotlib.pyplot as plt

######################################################
# LOAD THE HISTORY 
history_file = 'History/History_hidden_100_dropout_1.00_batch_32_l2_0.0000_lr_0.500_momentum_0.000_activation_sigmoid_epochs_500.pkl'
history = cp.load(open(history_file))
#######################################################

#######################################################
# Plot the Train and Val Error vs epochs
train_loss, = plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label="train_loss")
val_loss, = plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label="val_loss")
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
train_acc, = plt.plot(range(1, len(history['train_acc']) + 1), 1. - np.array(history['train_acc']), label="train_acc")
val_acc, = plt.plot(range(1, len(history['val_acc']) + 1), 1. - np.array(history['val_acc']), label="val_acc")
axes = plt.gca()
axes.set_ylim([0., .5])
plt.xlabel('Epochs')
plt.ylabel('Classification Error')
plt.title('Classification Error vs Epochs')
plt.legend([train_acc, val_acc], ["Training Classification Loss", "Validation Classification Loss"])
plt.savefig('Plots/Classification_loss.png')
plt.close()
########################################################