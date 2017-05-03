from itertools import izip

import numpy as np

from classifier import Classifier

class Perceptron(Classifier):

   # main function will convert data_with_labels from a set of 
	# ndarrays to a list of (label, python list of features) tuples
	def __init__(self, data_with_labels):
		assert len(data_with_labels) > 0

		self.training_set = data_with_labels
		self.labels = set([label for label, features in data_with_labels])
		
		# Initialize weights and biases
		feature_size = len(data_with_labels[0][1])
		self.weight_set = {}
		self.bias = {}
		for label in self.labels:
			self.weight_set[label] = [0 for i in range(self.feature_size)]
			self.bias[label] = 0

		self.train()

	def train(self):
		iterations = 0
		convergence = True
		while True:
			best_guess = "" 
			for image_label, image_features in self.training_set:
				best_guess = guessClassification(image_features) 
				if best_guess != image_label:
					self.update(image_features, image_label, best_guess)
					convergence = False
			iterations += 1
			if convergence or iterations > self.max_iterations:
				break

	def guessClassification(self, features):
		guesses = {}
		for label in self.weight_set:
			weights = self.weight_set[label]
			bias = self.bias[label]
			guesses[label] = reduce(lambda a, (w, f): a + w*f, izip(weights, image_features), 0) + bias
		return self.selectBestGuess(guesses)

	def update(self, feature_set, correct_label, incorrect_guess):
		# Decrease the weights for the incorrect guess for this image's features
		for i in range(len(self.weight_set[incorrect_guess])):
			self.weights[i] += -1 * feature_set[i]
		self.bias[incorrect_guess] += -1
		# Increase the weights for the correct classification for this image's features
		for i in range(len(self.weight_set[correct_label])):
			self.weights[i] += feature_set[i]
		self.bias[correct_label] += 1

	def selectBestGuess(self, d):
		v = list(d.values())
		k = list(d.keys())
		return k[v.index(max(v))]

	def classifyData(self, data):
		return guessClassification(self, data) 
