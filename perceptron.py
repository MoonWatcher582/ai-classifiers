from itertools import izip

import numpy as np

from classifier import Classifier
import utils

class Perceptron(Classifier):

	def __init__(self, data_with_labels):
		self.data_points = None
		self.weight_set = {}
		self.bias ={}

		# Set of correct classifications for training set
		# Indexes must correspond to the image in the features matrix 
		self.classification_set = None

	def train(self):
		iterations = 0
		convergence = True
		while True:
			img_idx = 0
			guesses = {}
			best_guess = "" 
			for image_features in self.image_matrix:
				for classification, weights in self.weight_set:
					guesses[classification] = reduce(lambda a, (w, f): a + w*f, izip(weights, image_features), 0) + self.bias[classification]
				best_guess = utils.selectBestGuess(guesses)
				if best_guess != self.classification_set[img_idx]:
					self.update(image_features, self.classification_set[img_idx], best_guess)
					convergence = False
				img_idx += 1
			iterations += 1
			if convergence or iterations > self.max_iterations:
				break

	def update(self, feature_set, correct_classification, incorrect_guess):
		# Decrease the weights for the incorrect guess for this image's features
		for i in range(len(self.weight_set[incorrect_guess])):
			self.weights[i] += -1 * feature_set[i]
		self.bias[incorrect_guess] += -1
		# Increase the weights for the correct classification for this image's features
		for i in range(len(self.weight_set[correct_classification])):
			self.weights[i] += feature_set[i]
		self.bias[correct_classification] += 1

	def classifyData(self, data):
		pass
