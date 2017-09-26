############ Running the Code ##############
python run_net.py 
Arguments
-train_file <TRAIN FILE PATH> : The training file location
-val_path <VAL FILE PATH> : The validation file location
-n_hidden <N_HIDDEN> : The number of hidden units (for a one layer network)
-batch <BATCH SIZE> : The batch size
-l2 <L2> : The L2 regularization value
-lr <LR> : The Learning Rate
-momentum <Gamma> : The momentum parameter
-activation <Activation> : The non linearity. Supported sigmoid, tanh, relu, leaky_relu
-n_epochs <EPOCHS> : Number of epochs
-save_prefix : The prefix with which models are saved

##############################################
Constructing network

Passing different layers as list elements to the layer_info parameters of neural_net constructs the network
Eg:

layer_info = [("hidden", 100, "sigmoid", 0.9), ("output", 10, "softmax", 1.)]
Generates a one layer network with sigmoid activation and a dropout of 0.9

layer_info = [("hidden", 100, "tanh", .5), ("batchnorm", 100), ("hidden", 100, "tanh", .5), ("batchnorm", 100), ("hidden", 100, "relu", 0.5), ("output", 10, "softmax", 1.)]
Generates a 3 layer network with tanh non-linearity for the first two layers, and relu non-lineariry for the third, and with two batchnorm layers.
################################################
Adding activations 
Just defining the activation and activation_grad in activations.py should be sufficient for defining the non-linearity.
################################################
Other Caveats:
Run the model with 
a Data/ directory with all the data files
a History/ directory to store the history
a Parameters/ directory to store the parameters

Model history saves everything : Training Accuracy, Training Loss, Validation Accuracy, Validation Loss, Best Validation Accuracy achieved during the run. The histories are stored in History/
BatchNorm Layer uses a running exponential average to keep track of the population mean and variance
Dropout implemented is inverse dropout, as is standard practice
The model saves the best model encountered so far during the epoch in Parameters/
#################################################