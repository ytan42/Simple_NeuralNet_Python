# Simple_NeuralNet_Python
2 hidden layers neural net for regression

A very simple neural network coded in Python by using numpy and pandas only. The only other package used was scikitlearn for normalise
the toy dataset. 

It's intended for self study only and not for commercial useage or production system.

The model has two hidden layers but can be extended to accommodate more layers. The number of neurons for each hidden layer can be changed. The activation function for the two hidden layers is tanh. Since the model is for regression, the activation function for the output layer is just a linear combination of weights, output of previous hidden layer and biases. The code does not have training and validation step as I coded this for studying the neural network purpose purpose.
