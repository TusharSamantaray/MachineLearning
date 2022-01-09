A recurrent neural network (RNN) in the figure below can be used for time-series prediction. For instance, if an input signal x(t) is a noisy sine wave, the network can be trained to estimate the next value x(t+ 1) at its outputˆy(t).

![image-20220109183111196](https://github.com/TusharSamantaray/MachineLearning/blob/main/Recurrent%20Neural%20Network/RNN.PNG)

Implements the above network from scratch (i.e. without using libraries such as TensorFlow or PyTorch) including the training using gradient descent with backpropagation. Generate a mix of sine waves (e.g., 2Hz + 3Hz) that is 100 samples long and use it as the training signal. Feed the training signal into the network (either sample by sample or in minibatches of the desired size). The network should be trained to predict the next value of the signal. Use the following constraints: assume that weights U and W are fixed (V is the only trainable matrix of weights) such that all values of U are equal to 1 and values of W are chosen randomly (from distribution of your choice) to be between 0 and 1. Use an arbitrary number of neurons in the hidden layer. Assume x and y are one-dimensional. Use mean squared error as the loss function. Use a nonlinear activation function of your choice. Adjust the learning rate as you see fit.

Plot or print the loss as a function of training iteration. Briefly discuss the results and how some of the choices that you made impact them. Briefly discuss what could be done to improve the results.

Then test the network on a noisy input signal. Add noise of your choice to signal and briefly comment on its impact on the prediction.
