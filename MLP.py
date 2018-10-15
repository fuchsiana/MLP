import numpy as np

class MultiLayerPerceptron:
    
    """

    Parameters
    ----------
    hidden_units: int, optional (default = 8)
        Number of units/neurons in the hidden layer
        
    n_outputs: int, optional (default = 1)
        Number of output units/neurons in the final layer
        
    epochs = int, optional (default = 1000)
        Number of epochs in training
        
    learning_rate: float, optional (default = 0.01)
        Learning-rate to multiplty deltas by when adjusting weights
        
    regression: Boolean, optional (default = False) 
        If true, regression model mode is on; predict method produces float values for outputs.
        If false, classifier model mode is on.  If one output unit/neuron is selected, predict method will deliver 
        1 if output is >=0.5 and 0 otherwise.  If two or more output units/neurons are selected, predict method will
        return unit with highest value.
        
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
        
        


    """
    
    # Constructor for the classifier object
    def __init__(self, hidden_units = 8, n_outputs = 1, epochs = 1000, learning_rate = 0.01, regression = False, 
                 batch_size = 16, weights_range = 'norm_med', sigmoid_output = False, random_seed = None, verbose = 0):

        self.hidden_units = hidden_units
        self.n_outputs = n_outputs
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regression = regression        
        self.batch_size = batch_size
        self.weights_range = weights_range
        self.sigmoid_output = sigmoid_output
        self.random_seed = random_seed
        self.verbose = verbose
        
        
    def sigmoid(self, n):
        
        return 1 / (1 + np.exp(-n))
    
    
    def sigmoid_derivative(self, n):
        
        return n * (1 - n)
    
    
    def shuffler(self, X, y):
        # shuffle dataset to try to improve mini-batch performance
        # code adapted from https://play.pixelblaster.ro/blog/2017/01/20/how-to-shuffle-two-arrays-to-the-same-order/
        shuffler = np.arange(X.shape[0])
        np.random.shuffle(shuffler)
        X = X[shuffler]
        y = y[shuffler]
        
        return X, y
        
        
    def forward_pass(self, inputs):
        #input to hidden
        z1 = inputs.dot(self.Weights1)
        hidden_a = self.sigmoid(z1)
        
        #hidden to output
        z2 = hidden_a.dot(self.Weights2)
        if self.sigmoid_output:
            output_a = self.sigmoid(z2)
        else:
            output_a = z2
            
        return z1, hidden_a, output_a
    
    
    def calculate_error(self, y, output_a, n_instances):

        mse = (np.sum((y - output_a)**2)) / (2 * n_instances)
        mae =  (np.sum(np.absolute((y - output_a)))) / n_instances
        # adapted from https://maviccprp.github.io/a-neural-network-from-scratch-in-just-a-few-lines-of-python-code/
        output_a_error = ((1 / 2) * (np.power((y - output_a), 2)))

        return mse, mae, output_a_error
    
    
    def backwards_pass(self, inputs, true_output, z1, hidden_a, output_a):
        # output to hidden
        d_output_error = output_a - true_output
        if self.sigmoid_output:
            d_z2 = self.sigmoid_derivative(output_a)          
        else:
            d_z2 = 1
        d_output = d_output_error * d_z2
        d_Weights2 = np.transpose(hidden_a).dot(d_output_error) / self.batch_size  
        
        # hidden to input
        d_z1 = self.sigmoid_derivative(hidden_a)
        # the below line is causing problems when n_outputs > 1
        d_hidden_a = d_output.dot(np.transpose(self.Weights2)) * d_z1
        d_Weights1 = np.transpose(inputs).dot(d_hidden_a) / self.batch_size
        
        return d_Weights2, d_Weights1

    
    
    def update_weights(self, d_Weights2, d_Weights1, n_inputs):
        self.Weights1 -= self.learning_rate * d_Weights1
        self.Weights2 -= self.learning_rate * d_Weights2

        d_Weights2 = np.zeros((self.hidden_units, self.n_outputs))
        d_Weights1 = np.zeros((n_inputs, self.hidden_units))

        
    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. 
        y : array-like, shape = [n_samples] 
            The target values (class labels) as integers or strings.            
        
        
        Returns
        -------

        
        self : object
    
        """
        #set random seed
        np.random.seed(self.random_seed)
            
        X = np.asarray(X)
        y = np.asarray(y)
        if self.verbose > 1:
            print("X is: \n", X, "\n and y is \n", y)
            print("X shape is: \n", X.shape, "\n and y shape is \n", y.shape)
        
        n_inputs = X.shape[1]
        n_instances = X.shape[0]

        if self.batch_size > n_instances or self.batch_size < 1:
            self.batch_size = n_instances
            
        if self.verbose > 1:
            print("n_inputs is: \n", n_inputs, "\n and n_instances is \n", n_instances)
            print("self.batch_size is: \n", self.batch_size)
            
        # initialising network elements
        
        hidden_a = np.ones((self.hidden_units,1))
        output_a = np.ones((self.n_outputs ,1))
            
        if self.weights_range == 'norm_med':
            self.Weights1 = np.random.normal(loc=0.0, scale=0.3, size=(n_inputs, self.hidden_units))
            self.Weights2 = np.random.normal(loc=0.0, scale=0.3, size=(self.hidden_units, self.n_outputs))
        elif self.weights_range == 'norm_tiny':
            self.Weights1 = np.random.normal(loc=0.0, scale=0.1, size=(n_inputs, self.hidden_units))
            self.Weights2 = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_units, self.n_outputs))
        elif self.weights_range == 'uniform':
            self.Weights1 = np.random.uniform(low=-0.999, high=0.999, size=(n_inputs, self.hidden_units))
            self.Weights2 = np.random.uniform(low=-0.999, high=0.999, size=(self.hidden_units, self.n_outputs))
        # adapted from https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e    
        elif self.weights_range == 'xavier':
            self.Weights1 = np.random.randn(n_inputs, self.hidden_units) * np.sqrt(1 / n_inputs)
            self.Weights2 = np.random.randn(self.hidden_units, self.n_outputs) * np.sqrt(1 / self.hidden_units)
        else: # default norm_med
            self.Weights1 = np.random.normal(loc=0.0, scale=0.3, size=(n_inputs, self.hidden_units))
            self.Weights2 = np.random.normal(loc=0.0, scale=0.3, size=(self.hidden_units, self.n_outputs))
            
        if self.verbose > 1:
            print("Initial Weights1 shape: \n", self.Weights1.shape, "\n Initial Weights1 contents: \n", self.Weights1)
            print("Initial Weights2 shape: \n", self.Weights2.shape, "\n Initial Weights2 contents: \n", self.Weights2)

        d_Weights2 = np.zeros((self.hidden_units, self.n_outputs))
        d_Weights1 = np.zeros((n_inputs, self.hidden_units))
                
        # mini-batching structure adapted from https://suvojitmanna.online/2017-09-24-neural-net-from-scratch/
        # epoch loop
        for i in range(self.epochs):  
            # shuffle data each epoch
            X, y = self.shuffler(X, y)
            if self.verbose > 1:
                print("In epoch", i+1, "\n X is: \n", X, "\n and y is: \n", y)
            # initialise collection of epoch output data
            epoch_output = np.zeros((0,1))
            
            # mini-batch loop
            for j in range (0, n_instances, self.batch_size):
                count = 1
                mB_split = min(j + self.batch_size, n_instances)               
                X_mB = X[j:mB_split, :]
                y_mB = y[j:mB_split, :]
                if self.verbose > 1:
                    print("Epoch", i+1, ": mini-batch phase", count, ": mB_split position is:", mB_split)
                    print("X_mB shape: \n", X_mB.shape, "\n X_mB contents: \n", X_mB)
                    print("y_mB shape: \n", y_mB.shape, "\n y_mB contents: \n", y_mB)
                    
                
                # FeedForward 
                z1, hidden_a, output_a = self.forward_pass(X_mB)
                if self.verbose > 1:
                    print("After forward pass")
                    print("z1 shape: \n", z1.shape, "\n z1 contents: \n", z1)
                    print("hidden_a shape: \n", hidden_a.shape, "\n hidden_a contents: \n", hidden_a)
                    print("output_a shape: \n", hidden_a.shape, "\n output_a contents: \n", output_a)
                
                # update epoch_output
                epoch_output = np.r_[epoch_output, output_a]
#                 np.concatenate((epoch_output, output_a),axis=0)
                if self.verbose > 1:
                    print("epoch_output shape: \n", hidden_a.shape, "\n epoch_output contents: \n", hidden_a)                    
                
                if self.verbose > 1 and self.batch_size > n_instances:
                    mB_size = min(self.batch_size, n_instances - j)
                    mse, mae, output_a_error = self.calculate_error(y_mB, output_a, mB_size)
                    print("MSE on mini-batch", count, "in epoch", i+1, "is", mse)
                    print("mae on mini-batch", count, "in epoch", i+1, "is", mae)
                    print("output_a_error on mini-batch", count, "in epoch", i+1, "is", output_a_error)
                    
                # BackPropagation
                d_Weights2, d_Weights1 = self.backwards_pass(X_mB, y_mB, z1, hidden_a, output_a)
                
                if self.verbose > 1:
                    print("After backprop")
                    print("d_Weights2 shape: \n", d_Weights2.shape, "\n d_Weights2 contents: \n", d_Weights2)
                    print("d_Weights1 shape: \n", d_Weights1.shape, "\n d_Weights1 contents: \n", d_Weights1)             
                
                # update weights
                self.update_weights(d_Weights2, d_Weights1, n_inputs)
                if self.verbose > 1:
                    print("After update_weights")
                    print("New Weights1 shape: \n", self.Weights1.shape, "\n New Weights1 contents: \n", self.Weights1)
                    print("New Weights2 shape: \n", self.Weights2.shape, "\n New Weights2 contents: \n", self.Weights2)
                
                count +=1
            
            # calculate and display epoch metrics
            if self.verbose > 0:
                if self.verbose > 1:
                    print("Params for epoch weight calculation:")
                    print("y shape: \n", y.shape, "\n y contents: \n", y)
                    print("epoch_output shape: \n", epoch_output.shape, "\n epoch_output contents: \n", epoch_output)
                    print("n_instances: \n", n_instances)
                mse, mae, output_a_error = self.calculate_error(y, epoch_output, n_instances)
                if self.regression:
                    print("MSE for epoch", i+1, "is", mse, "\t MAE is", mae)
                    if self.verbose > 1:                    
                        print("output_a_error for epoch", i+1, "is", output_a_error)
                else:
                    print("MSE for epoch", i+1, "is", mse, "\t MAE is", mae)
                    if self.verbose > 1:                    
                        print("output_a_error for epoch", i+1, "is", output_a_error)
            else:
                if (i+1) % (self.epochs/20) == 0:
                    mse, mae, output_a_error = self.calculate_error(y, epoch_output, n_instances)
                    print("MSE for epoch", i+1, "is", mse, "\t MAE is", mae)

        # Return the classifier
        return self

    
    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        """Predict class labels of the input samples X.
        
        Parameters
        ----------
        X : array-like matrix of shape = [n_instances, n_features]
            The input samples. 
        
        Returns
        -------
        
        One of the following:
        
        output_a : array of shape = [n_instances, ].
            The predicted output values for the input samples. 
        output_b : an binary number
            The prediction of true or false for a binary output.  
        output_c : an array of shape = [n_instances, ].
            The predicted class labels of the input samples
        
        """
        z1, hidden_a, output_a = self.forward_pass(X)
        
        if self.regression:
            
            output_acts = np.array(output_a) #[:, np.newaxis]
            return output_a
        
        elif self.n_outputs > 1:


            print("Prediction probabilities: \n", output_a)           
            output_c = np.array(output_a) #[:, np.newaxis]
            return output_c
            
        else:
            
            prediction = [1 if i >= 0.5 else 0 for i in output_a]
            print("Prediction probabilities: \n", output_a)
            output_b = np.array(prediction)[:, np.newaxis]

            return output_b
