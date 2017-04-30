from classifier import Classifier
from itertools import izip
import numpy as np

class Perceptron(Classifier):

	def __init__(self, data_with_labels):
		self.data_points = None
		self.weights = None
		self.bias = 0

		# Set of correct classifications for training set
		# Indexes must correspond to the image in the features matrix 
		self.classification_set = None

	def train(self):
		iterations = 0
		convergence = True
		while True:
			img_idx = 0
			for image_features in self.image_matrix:
				guess = np.sign(reduce(lambda a, (w, f): a + w*f, izip(self.weights, image_features), 0) + self.bias)
				if guess == self.classification_set[img_idx]:
					self.update(image_features, self.classification_set[img_idx])
					convergence = False
				img_idx += 1
			iterations += 1
			if convergence or iterations > self.max_iterations:
				break

	def update(self, feature_set, image_classification):
		for i in xrange(self.weights):
			self.weights[i] += image_classification * feature_set[i]
		self.bias += image_classification

	def classifyData(self, data):
		pass
