# MLP
Implementation of a Multi-Layer Perceptron in Python

MLP.py contains the class MultiLayerPerceptron(), a functioning MLP Regressor.

MultiLayerPerceptron() parameters include:

    hidden_units: int, optional (default = 8)
        Number of units/neurons in the hidden layer
        
    epochs = int, optional (default = 1000)
        Number of epochs in training
        
    learning_rate: float, optional (default = 0.01)
        Learning-rate to multiplty deltas by when adjusting weights
        
    batch_size: int, optional (default = 16) 
        Size of minibatches; set as 1 for SGD and as input size for full batch mode
    
    weights_range: string, optional (default = rand_med) 
        Range of values weights to be selected from.  Available options are:
        norm_med : returns normally distibuted random values with a mean of 0.0 and a standard distribution of 0.3 
        norm_tiny : returns normally distibuted random values with a mean of 0.0 and a standard distribution of 0.1
        uniform : returns uniformly distibuted random values between (almost) -1 and 1 
        xavier : returns random range in a range defined by the Xanier method
        
    sigmoid_output: Boolean, optional (default = False)
        If true, sigmoid activation performed on output neuron(s)
        If false, no/linear activation performed on output neuron(s)
        
    random_seed: int, optional (default = None)
        Set as int for replicable results; leave as default 'None' for random.
        
     verbose: int, optional (default = 0)
         If 0, error calculations printed out after every 5% of data processed
         If 1, error calculations printed out after every mini-batch
         If 2, EVERYTHING printed out (for debugging use)
         
MultiLayerPerceptron() methods are:
         
    fit(X, y), to train a model, with parameters:
            X : array-like, shape = [n_samples, n_features]
                The training input samples. 
            y : array-like, shape = [n_samples] 
                The target values (class labels) as integers or strings.            
        Returns object - the trained model
            
    predict(X), to predict class labels of the input samples X, with parameter:
            X : array-like matrix of shape = [n_instances, n_features]
                The input samples. 
        Returns array of shape = [n_instances, ]. The predicted output values for the input samples. 

         
To do:
 * Implement classification
 * Implement option to choose multiple hidden layers
 * Implement greater range of activation options
 * Apply to some more interesting data
