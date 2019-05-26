# Simple_NeuralNet_Python
A simple fully connected neural network with 4 layers - 1 input, 2 hidden and 1 output for regression purpose.

Primarily coded in Python by using numpy and pandas. The only other package used was scikitlearn for the normalise function. 

It's intended for self study only and not for commercial useage or production system.

The model has two hidden layers but can be extended to accommodate more layers. The number of neurons for each hidden layer can be changed. The activation function for the two hidden layers is tanh. Since the model is for regression, the activation function for the output layer is just a linear combination of weights, output of previous hidden layer and biases. The code does not have training and validation step as I coded this for studying the neural network purpose purpose.
