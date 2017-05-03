from collections import Counter
from itertools import izip
import operator
import random

import numpy as np

# An interface for all classifiers to implement.
class Classifier():

    # Training should happen in classifier initialization.
    # data_with_labels should be a list of tuples in the form of
    # (label, <feature vector>). All feature vectors *MUST* be the same length.
    def __init__(self, data_with_labels):
        raise NotImplementedError("init is not implemented")

    # This method should be used to classify a single data point. the data
    # point should be a feature vector.
    def classifyData(self, data):
        raise NotImplementedError("classifyData is not implemented")


# A Naive Bayes classifer. Implements Classifer.
class NaiveBayesClassifier(Classifier):

    def __init__(self, data_with_labels):
        # If we have no data, we don't know anything about the features.
        assert len(data_with_labels) > 0

        self.min_probability = 0.5
        self.labels = []

        # This is a list of maps. Each entry in the list for the given feature
        # and the map gives the probabilty of the feature's value given the
        # different values of the label.
        self.feature_probabilities = []

        # This is in the same format as above, but is the counts instead of
        # probabilities.
        self.feature_counts = []

        for feat_num in range(0, len(data_with_labels[0][1])):
            # Count how many times each value of the feature has shown up for
            # each label.
            self.feature_counts.append(dict())
            feature_map = self.feature_counts[feat_num]
            for data in data_with_labels:
                label = data[0]
                feature = data[1][feat_num]
                if feature_map.get(label) == None:
                    feature_map[label] = dict()
                if feature_map.get(label).get(feature) == None:
                    feature_map[label][feature] = 0
                feature_map[label][feature] += 1

            # Convert counts to probabilities.
            self.feature_probabilities.append(dict())
            prob_map = self.feature_probabilities[feat_num]
            for label, value_map in feature_map.iteritems():
                self.labels.append(label)
                prob_map[label] = dict()
                total_count = 0
                for value, count in value_map.iteritems():
                    total_count += count
                for value, count in value_map.iteritems():
                    probability = float(count)/total_count
                    prob_map[label][value] = probability
                    if probability / 2 < self.min_probability:
                        self.min_probability = probability / 2


    def classifyData(self, data):
        probabilities = []
        for label in self.labels:
            # Calculate the probabilty of each label given the observed
            # features.
            label_prob = 1
            feat_num = 0
            for feature in data:
                feature_prob = self.feature_probabilities[feat_num].get(label).get(feature)
                if feature_prob == None:
                    feature_prob = self.min_probability
                label_prob *= feature_prob
                feat_num += 1
            probabilities.append(label_prob)

        # Return the label with the maximum probability.
        return self.labels[probabilities.index(max(probabilities))]


class RandomForestClassifier(Classifier):

    class DecisionTree:
        def __init__(self, feat_num):
            self.labels = []
            self.feature = feat_num
            self.children = dict()

    def __init__(self, data_with_labels):
        random.seed()
        self.decision_trees = []

        t2 = self.DecisionTree(1)
        t2.children[0] = self.DecisionTree(2)
        t2.children[0].labels.append(("foo", 0.5))
        t2.children[0].labels.append(("bar", 0.5))
        self.decision_trees.append(t2)

        t1 = self.DecisionTree(0)
        t1.children[0] = self.DecisionTree(1)
        t1.children[0].labels.append(("baz", 1.0))
        self.decision_trees.append(t1)


    def classifyData(self, data):
        # Check through each decision tree to get a value, and pick the mode of
        # the values returned.
        decisions = dict()
        for tree in self.decision_trees:
            for i in range (0, len(data)):
                assert tree.feature >= i
                if tree.feature < i:
                    continue
                tree = tree.children.get(data[i])
                if tree == None:
                    break
                if len(tree.labels) > 0:
                    break
            if tree == None or len(tree.labels) == 0:
                continue
            r = random.random()
            i = 1.0
            label = None
            for tup in tree.labels:
                i -= tup[1]
                if i <= r:
                    label = tup[0]
                    break
            if decisions.get(label) == None:
                decisions[label] = 0
            decisions[label] += 1

        return max(decisions.iteritems(), key=operator.itemgetter(1))[0]


class Perceptron(Classifier):

    # main function will convert data_with_labels from a set of 
    # ndarrays to a list of (label, python list of features) tuples
    def __init__(self, data_with_labels):
        assert len(data_with_labels) > 0

        self.max_iterations = 50

        self.training_set = data_with_labels
        self.labels = set([label for label, features in data_with_labels])
		
        # Initialize weights and biases
        feature_size = len(data_with_labels[0][1])
        self.weight_set = {}
        self.bias = {}
        for label in self.labels:
            self.weight_set[label] = [0 for i in range(feature_size)]
            self.bias[label] = 0

        iterations = 0
        convergence = True
        while True:
            best_guess = "" 
            for image_label, image_features in self.training_set:
                best_guess = self.guessClassification(image_features) 
                if best_guess != image_label:
                    self.update(image_features, image_label, best_guess)
                    convergence = False
            iterations += 1
            if convergence or iterations > self.max_iterations:
                break


    def classifyData(self, data):
        return self.guessClassification(data) 


    def guessClassification(self, features):
        guesses = {}
        for label in self.weight_set:
            weights = self.weight_set[label]
            bias = self.bias[label]
            guesses[label] = reduce(lambda a, (w, f): a + w*f, izip(weights, features), 0) + bias
        return self.selectBestGuess(guesses)


    def update(self, feature_set, correct_label, incorrect_guess):
        # Decrease the weights for the incorrect guess for this image's features
        for i in range(len(self.weight_set[incorrect_guess])):
            self.weight_set[incorrect_guess][i] += -1 * feature_set[i]
        self.bias[incorrect_guess] += -1
        # Increase the weights for the correct classification for this image's
        # features
        for i in range(len(self.weight_set[correct_label])):
            self.weight_set[correct_label][i] += feature_set[i]
        self.bias[correct_label] += 1


    def selectBestGuess(self, d):
        v = list(d.values())
        k = list(d.keys())
        return k[v.index(max(v))]


class KNearestNeighbors(Classifier):

    def __init__(self, data_with_labels):
        self.k = 3
        assert len(data_with_labels) > self.k

        self.label_set = [label for label, features in data_with_labels]
        self.feature_set = [features for label, features in data_with_labels]

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
        return count.most_common(1)[0][0]


    def selectKSmallest(self, values):
        # Return the labels with same indicies as the k smallest values 
        A = np.array(values)
        indicies = np.argpartition(A, self.k)
        return [self.label_set[i] for i in indicies]

def main():
    # TODO: Feature extraction here
    # TODO: Train classifier here.
    # TODO: Classify test data here.
    data_with_labels = [
            (0, [0,0]),
            (1, [2,2]),
            (0, [1,1]),
            (0, [0,2]),
				(1, [2,3]),
    ]
    c = NaiveBayesClassifier(data_with_labels)
    print c.classifyData([0,2])

    c = Perceptron(data_with_labels)
    print c.classifyData([0,2])

    c = KNearestNeighbors(data_with_labels)
    print c.classifyData([0,2])

    # Simple random forest test.
    c = RandomForestClassifier(None)
    for i in range(0, 10):
        print c.classifyData([0, 0])

	

if __name__ == '__main__':
    main()
