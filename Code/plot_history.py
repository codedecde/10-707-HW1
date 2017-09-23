import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle as cp

####### LOAD THE HISTORY ##########
history_file = 'History/History_hidden_100_dropout_1.00_batch_32_l2_0.0000_lr_0.100_momentum_0.000_activation_sigmoid_epochs_50.pkl'
history = cp.load(open(history_file))
###################################

####### Plot the Train and Val Error vs epochs ########
train_loss, = plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label="train_loss")
val_loss, = plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label="val_loss")
axes = plt.gca()
axes.set_ylim([0., 3.])
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Error')
plt.title('Entropy Error vs Epochs')
plt.legend([train_loss, val_loss], ["Training Loss", "Validation Loss"])
plt.savefig('Plots/Entropy_loss.png')
