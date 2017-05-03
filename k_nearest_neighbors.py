from collections import Counter
import numpy as np

from classifier import Classifier

class KNearestNeighbors(Classifier):

	def __init__(self, data_with_labels):
		self.label_set = [label for label, features in data_with_labels]
		self.feature_set = [features for label, features in data_with_labels]
		self.k = 1

		self.label_types = set(self.label_set)

	def classifyData(self, data):
		# Subtract training data from the data, pairwise
		dx = [np.subtract(data, train) for train in self.feature_set]
		# Compute square sum for each difference vector
		ssd = [reduce(lambda a, i: a + i**2, x, 0) for x in dx]
		# Select the label with most represented by the k-min values
		k_smallest = self.selectKSmallest(ssd)
		# return the label that appears the most  
		count = Counter(k_smallest)
# TODO: what do if tie?
		return count.most_common(1)[0][0]

	def selectKSmallest(self, values):
		# Return the labels with same indicies as the k smallest values 
		A = np.array(values)
		indicies = np.argpartition(A, self.k)
		return [self.label_set[i] for i in indicies]

def main():
	data_with_labels = [
		(0, [0,0]),
		(1, [1,1]),
		(0, [1,1]),
		(0, [0,2]),
	]
	c = KNearestNeighbors(data_with_labels)
	print c.classifyData([0,2])

if __name__ == '__main__':
	main()
