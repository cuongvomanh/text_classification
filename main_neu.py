import os
import numpy as np
import pickle
import random
import my_lib
import matplotlib.pyplot as plt
import math
# path='C:/Users/user/Documents/machine_learning_btl2/preprocessing/data'
# train_input_path=path+'/train_input.txt'
# train_lable_path=path+'/train_lable.txt'
# test_input_path=path+'/test_input.txt'
# test_lable_path=path+'/test_lable.txt'
# train_input_list=my_lib.read_input(train_input_path)
# train_lable_list=my_lib.read_lable(train_lable_path)
# train_lable_input_list=[[input,lable]for input,lable in zip(train_input_list,train_lable_list)]

# test_input_list=my_lib.read_input(test_input_path)
# test_lable_list=my_lib.read_lable(test_lable_path)
# test_lable_input_list=[[input,lable]for input,lable in zip(test_input_list,test_lable_list)]

# dimesion_number= len(train_input_list[0])
# train_input_list=None
# train_lable_list=None
# test_input_list=None
# test_lable_list=None
subject_number=20
source_file_path='C:/Users/user/Documents/machine_learning_btl2/preprocessing/data/data.txt'
# source_file_path='C:/Users/user/Documents/machine_learning_btl2/preprocessing/list_vector.txt'
train_lable_input_list,test_lable_input_list=my_lib.tranform_from_sparse_matrix_to_dense_matrix(source_file_path,subject_number)
dimesion_number=len(train_lable_input_list[0][0])
print('1')


class Network(object):

	def __init__(self, sizes):
		"""The list ``sizes`` contains the number of neurons in the
		respective layers of the network.  For example, if the list
		was [2, 3, 1] then it would be a three-layer network, with the
		first layer containing 2 neurons, the second layer 3 neurons,
		and the third layer 1 neuron.  The biases and weights for the
		network are initialized randomly, using a Gaussian
		distribution with mean 0, and variance 1.  Note that the first
		layer is assumed to be an input layer, and by convention we
		won't set any biases for those neurons, since biases are only
		ever used in computing the outputs from later layers."""
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x)
						for x, y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self, a):
		"""Return the output of the network if ``a`` is input."""
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a)+b)
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta,
			test_data=None):
		"""Train the neural network using mini-batch stochastic
		gradient descent.  The ``training_data`` is a list of tuples
		``(x, y)`` representing the training inputs and the desired
		outputs.  The other non-optional parameters are
		self-explanatory.  If ``test_data`` is provided then the
		network will be evaluated against the test data after each
		epoch, and partial progress printed out.  This is useful for
		tracking progress, but slows things down substantially."""
		accurancy_list=[]
		index_list=[]
		if test_data: n_test = len(test_data)
		n = len(training_data)
		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)
			accurancy=float(self.evaluate(test_data)) / n_test *100
			accurancy_list.append(accurancy)
			index_list.append(j)
			if j%1==0:
				if test_data:
					print ("Epoch {0}: {1} ".format(j, accurancy))
				else:
					print ("Epoch {0} complete".format(j))
			if accurancy>95:
				break
		plt.plot(index_list,accurancy_list)
		plt.axis([0, epochs, 0, 100])
		plt.xlabel('epoch')
		plt.ylabel('accurancy')
		plt.show()
	def update_mini_batch(self, mini_batch, eta):
		"""Update the network's weights and biases by applying
		gradient descent using backpropagation to a single mini batch.
		The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
		is the learning rate."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w-(eta/len(mini_batch))*nw
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb
					   for b, nb in zip(self.biases, nabla_b)]

	def backprop(self, x, y):
		"""Return a tuple ``(nabla_b, nabla_w)`` representing the
		gradient for the cost function C_x.  ``nabla_b`` and
		``nabla_w`` are layer-by-layer lists of numpy arrays, similar
		to ``self.biases`` and ``self.weights``."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# feedforward
		activation = x
		activations = [x] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# backward pass
		delta = self.cost_derivative(activations[-1], y) * \
			sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		# Note that the variable l in the loop below is used a little
		# differently to the notation in Chapter 2 of the book.  Here,
		# l = 1 means the last layer of neurons, l = 2 is the
		# second-last layer, and so on.  It's a renumbering of the
		# scheme in the book, used here to take advantage of the fact
		# that Python can use negative indices in lists.
		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def evaluate(self, test_data):
		"""Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""
		test_results = [(np.argmax(self.feedforward(x)), (np.argmax(y)) ) for (x, y) in test_data]
		# print(shape())
		# x,y=test_results[1]
		# print(x,y)
		# print(test_results[1])
		return sum(np.int(x == y) for (x, y) in test_results)

	def cost_derivative(self, output_activations, y):
		"""Return the vector of partial derivatives \partial C_x /
		\partial a for the output activations."""
		return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
	"""The sigmoid function."""
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	"""Derivative of the sigmoid function."""
	return sigmoid(z)*(1-sigmoid(z))


size_layer_list=[dimesion_number,subject_number]
epoch_size=40
mini_batch_size=10
network=Network(size_layer_list)
# network.SGD(training_data, 30, 1, 3.0, test_data=test_data)
network.SGD(train_lable_input_list, epoch_size, mini_batch_size, 3.0, test_data=test_lable_input_list)
# net = network.Network([784, 30, 10])
# network.train(train_lable_input_list,test_lable_input_list,epoch_size,mini_batch_size)