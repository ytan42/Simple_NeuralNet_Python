# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

################################
### Helper functions
################################

# Weight initiation function
def init_weight(n: int, m: int) -> np.ndarray:
    weight_matrix = np.random.uniform(low = 0.0, high = 1.0, size = (n, m))
    return(weight_matrix)

# Bias initiation function
def init_bias(m: int) -> np.ndarray:
    bias_matrix = np.random.uniform(low = 0.0, high = 1.0, size = (1, m))
    return(bias_matrix)

# The sigmoid function
def sigmoid(z):
    return(1 / (1 + np.exp(z * -1)))

# The detrivative of sigmoid function
def sigmoid_d(z, sigmoid_z=None):
    # If sigmoid_z is directly available, then we don't need to calculate
    # it again
    if sigmoid_z is not None:
        return(sigmoid_z * (1 - sigmoid_z))
    else:
        return(sigmoid(z) * (1 - sigmoid(z)))

# The Hyperbolic Tangent function
def tanh(z):
    return((np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z)))
    
# The derivative of tanh function
def tanh_d(z, tanh_z=None):
    if tanh_z is not None:
        return(1-(tanh_z ** 2))
    else:
        return(1-(tanh(z) ** 2))

# Mean squared error function
def mean_sqr_err(y, y_hat):
    out = sum(((y - y_hat) ** 2)) / len(y)
    return(out)


def train_test_split(X, Y, train_size=1500):
    
    # Training and testing split
    train_idx = np.random.choice(np.arange(len(X)), size=train_size, replace=False)

    X_train = X[train_idx]
    X_test = X[~train_idx]

    # The reshape is to make it n by 1 matrix
    y_train = Y[train_idx].reshape(-1, 1)
    y_test = Y[~train_idx].reshape(-1, 1)

    out = {'X_train' : X_train, 
           'X_test' : X_test, 
           'y_train' : y_train, 
           'y_test' : y_test}

    return(out)


def nn_compare(X_train, y_train, X_test):

    clf = MLPRegressor(activation='tanh', solver='sgd', batch_size=20
                       , learning_rate_init=0.005, max_iter = 50
                       , hidden_layer_sizes=(10, 10))
    regr = clf.fit(X_train, y_train)

    y_hat = regr.predict(X_test)

    return y_hat


################################
### Neural NetworK
################################

# Notation:
# xi - the ith batch of data
# w1 - weights between input and 2nd layer
# b1 - biases of 2nd layer
# a2 - output of 2nd layer
# w2 - weights between 2nd and 3rd
# b2 - biases of 3rd layer
# a3 - output of 3rd layer
# w3 - weights between 3rd and output layer
# b3 - biases of output layer
# z4 - output of the network

# The feedforward function
def feedforward(xi, w1, b1, w2, b2, w3, b3):

  # Input layer - takes input from data
  a1 = xi

  # 2nd hidden layer
  z2 = np.dot(a1, w1) + b1
  a2 = tanh(z2)

  # 3rd hidden layer
  z3 = np.dot(a2, w2) + b2
  a3 = tanh(z3)

  # 4th and output layer. Only linear combination not activation function.
  z4 = np.dot(a3, w3) + b3

  # Output the prediction
  return({'w1' : w1,
          'z2' : z2,
          'w2' : w2,
          'z3' : z3,
          'w3' : w3,
          'z4' : z4,
          'a3' : a3,
          'a2' : a2,
          'a1' : a1})

# Backpropagation function (simple version)
def backpropagation(yi: np.array, forward: dict) -> dict: 
    
    # Unpack variables
    a2 = forward['a2'] # tanh of z2
    z2 = forward['z2']
    w2 = forward['w2']

    a3 = forward['a3'] # tanh of z3
    z3 = forward['z3']
    w3 = forward['w3']

    z4 = forward['z4']
    
    # Output layer error - the derivative of 1/2(y - y_hat)^2
    d4 = -(yi.reshape(len(yi), 1) - z4)
    
    # 3rd layer error
    d3 = np.dot(d4, w3.T) * tanh_d(z3, tanh_z=a3)
    
    # 2nd layer error
    d2 = np.dot(d3, w2.T) * tanh_d(z2, tanh_z=a2)
    
    return({'d4':d4, 
            'd3':d3, 
            'd2':d2})


# The prediction function is the feedforward
def predict_NN(X, model):
    
    # Unpack weights and biases from modell object
    w1 = model['w1']
    b1 = model['b1']  
    w2 = model['w2']
    b2 = model['b2'] 
    w3 = model['w3']
    b3 = model['b3']
    
    output = feedforward(xi=X, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
    
    return(output['z4'])

################################
### The main build function
################################

def build_NN(X: np.ndarray, 
             y: np.ndarray, 
             h1_neurons: int, 
             h2_neurons: int, 
             epoch: int, 
             eta: float, 
             batch_size: int) -> dict:
    
    np.random.seed(123)    

    # Initiate weights and biases
    X_rows, X_columns = X.shape

    w1 = init_weight(n = X_columns, m = h1_neurons)
    b1 = init_bias(m = h1_neurons)
    
    w2 = init_weight(n = h1_neurons, m = h2_neurons)
    b2 = init_bias(m = h2_neurons)
    
    w3 = init_weight(n = h2_neurons, m = 1)
    b3 = init_bias(m = 1)
    
    # Number of batches
    n_batch = int(X_rows / batch_size) 
    
    # To record error for each epoch
    error_epoch = dict() 
    
    for epoch in range(epoch):
        
        if epoch % 5 == 0:
            print(f"Processing {epoch}...")

        # Initiate weight and bias delta to matrix of 0s
        w3_delta = w3 * 0
        w2_delta = w2 * 0
        w1_delta = w1 * 0
        b3_delta = b3 * 0
        b2_delta = b2 * 0
        b1_delta = b1 * 0
        
        # idx_batches contains a matrix of randomly generated index number
        # where the number of row equals to the number of batches
        # and value contains index number
        idx_batches = np.random.choice(np.arange(X_rows), size=(n_batch, batch_size), replace=False)
        
        # Start stochastic batch gradient descent without replacement
        for batch in range(n_batch):
            
            # Processing the entire batch in one go
            xi = X[idx_batches[batch], :]
            yi = y[idx_batches[batch]]
            
            forward = feedforward(xi=xi, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
            
            backprop = backpropagation(yi, forward)
            
            # Cumulate weight deltas
            w3_delta = w3_delta + np.dot(forward['a3'].T, backprop['d4'])
            w2_delta = w2_delta + np.dot(forward['a2'].T, backprop['d3'])
            w1_delta = w1_delta + np.dot(forward['a1'].T, backprop['d2'])
            
            # Cumulate bias deltas
            b3_delta = b3_delta + backprop['d4'].sum()
            b2_delta = b2_delta + backprop['d3'].sum(axis=0).reshape(1, -1)
            b1_delta = b1_delta + backprop['d2'].sum(axis=0).reshape(1, -1)
                
            # Update weights and biases after finish one batch.
            # Then, use the updated weights and biases for the next batch.
            w1 = w1 - eta * (w1_delta / batch_size)
            w2 = w2 - eta * (w2_delta / batch_size)
            w3 = w3 - eta * (w3_delta / batch_size)

            b1 = b1 - eta * (b1_delta / batch_size)
            b2 = b2 - eta * (b2_delta / batch_size)
            b3 = b3 - eta * (b3_delta / batch_size)
        
        # At the each epoch, make prediction by using current weights and biases.
        epoch_predit = predict_NN(X, {'w1':w1, 'w2':w2, 'w3':w3, 'b1':b1, 'b2':b2, 'b3':b3})
        
        # Calculate mean sqr err so far for current epoch
        epoch_err = mean_sqr_err(y, epoch_predit)
        error_epoch[epoch] = epoch_err
        
    out = {'w1' : w1,
           'w2' : w2,
           'w3' : w3,
           'b1' : b1,
           'b2' : b2,
           'b3' : b3,
           'error_epoch' : error_epoch}
    
    return(out)


def main():

    # Generate a toy dataset to test the neural network
    X, Y = make_regression(n_samples=2000, n_features=10)

    train_test = train_test_split(X, Y, train_size=1500)

    X_train = train_test['X_train']
    y_train = train_test['y_train']
    
    X_test = train_test['X_test']
    y_test = train_test['y_test']

    # Standardise data by using sklearn's StandardScaler
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_trans = scaler_X.transform(X_train)
    y_train_trans = scaler_y.transform(y_train)

    # To make sure we are not cheating, use the scaler created by
    # the training set to standarise the test set
    X_test_trans = scaler_X.transform(X_test)

    # Call the build_NN function to build nerual network
    nn_out = build_NN(X_train_trans, 
                    y_train_trans, 
                    h1_neurons = 10, 
                    h2_neurons = 10, 
                    epoch = 50, 
                    eta = 0.005, 
                    batch_size = 20)

    # Use the NN to predict unseen data
    X_test_trans = scaler_X.transform(X_test)
    y_hat_self = predict_NN(X=X_test_trans, model=nn_out)

    # sklearn to predict
    y_hat_sklearn = nn_compare(X_train, y_train.ravel(), X_test)

    # Self-build NN performance
    self_perf = mean_sqr_err(y=y_test, y_hat=scaler_y.inverse_transform(y_hat_self))

    # sklearn NN performance
    sk_perf  = mean_sqr_err(y=y_test, y_hat=scaler_y.inverse_transform(y_hat_sklearn.reshape(-1, 1)))

    print(f'Self built NN performance: {self_perf}')
    print(f'SKlearn NN performance: {sk_perf}')


if __name__ == '__main__':
    main()

   
