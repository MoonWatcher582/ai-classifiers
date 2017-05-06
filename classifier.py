from __future__ import print_function
from collections import Counter
from itertools import izip

import math
import operator
import random
import feature_extract
import sys

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
                if label not in self.labels:
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
            label_prob = 1.0
            feat_num = 0
            for feature in data:
                feature_prob = self.feature_probabilities[feat_num].get(label).get(feature)
                if feature_prob == None:
                    feature_prob = self.min_probability
                if feature_prob < 1:
                    label_prob *= feature_prob
                feat_num += 1
            probabilities.append(label_prob)

        # Return the label with the maximum probability.
        return self.labels[probabilities.index(max(probabilities))]


# A random forest classifier.
class RandomForestClassifier(Classifier):

    # DecisionTree is a single decision tree node. It's children are contained
    # in a dictionary keyed by the feature value.
    class DecisionTree:
        def __init__(self, feat_num):
            self.labels = []
            self.feature = feat_num
            self.children = dict()

    def __init__(self, data_with_labels):
        random.seed()
        self.decision_trees = []

        assert len(data_with_labels) > 0

        #num_features = int(math.sqrt(len(data_with_labels[0][1])))
        num_features = 10
        num_trees = int(math.sqrt(len(data_with_labels[0][1])))
        for i in range(0, num_trees):
            tree = self.DecisionTree(0)
            feature_set = set()
            while len(feature_set) < num_features:
                feature_set.add(random.randint(0,
                    len(data_with_labels[0][1])-1))
            feature_list = sorted(feature_set)
            print("Building tree", i, "out of ", num_trees)
            self.decision_trees.append(self.build_tree(tree, feature_list,
                data_with_labels))

    def build_tree(self, tree, feature_list, data_with_labels):
        if len(feature_list) == 0:
            return tree
        tree.feature = feature_list[0]
        data_with_labels_per_value = dict()
        for data in data_with_labels:
            feature_val = data[1][feature_list[0]]
            if data_with_labels_per_value.get(feature_val) == None:
                data_with_labels_per_value[feature_val] = []
            data_with_labels_per_value[feature_val].append(data)
        for feature_val, data in data_with_labels_per_value.iteritems():
            tree.children[feature_val] = self.build_tree(self.DecisionTree(0),
                    feature_list[1:], data)
            if len(feature_list) == 1:
                label_counts = dict()
                for data_point in data:
                    if label_counts.get(data_point[0]) == None:
                        label_counts[data_point[0]] = 0
                    label_counts[data_point[0]] += 1
                for label, count in label_counts.iteritems():
                    tree.children[feature_val].labels.append((label,
                        float(count)/len(data)))
        return tree

    def classifyData(self, data):
        # Check through each decision tree to get a value, and pick the mode of
        # the values returned.
        decisions = dict()
        for tree in self.decision_trees:
            for i in range (0, len(data)):
                assert tree.feature >= i
                if tree.feature > i:
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

        if len(decisions) == 0:
            return None
        return max(decisions.iteritems(), key=operator.itemgetter(1))[0]


class Perceptron(Classifier):

    # main function will convert data_with_labels from a set of
    # ndarrays to a list of (label, python list of features) tuples
    def __init__(self, data_with_labels):
        assert len(data_with_labels) > 0

        self.max_iterations = 7 

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
            print("Perceptron training iteration " + str(iterations + 1) + "/" + str(self.max_iterations), end="\r")
            sys.stdout.flush()
            best_guess = ""
            for image_label, image_features in self.training_set:
                best_guess = self.guessClassification(image_features)
                if best_guess != image_label:
                    self.update(image_features, image_label, best_guess)
                    convergence = False
            iterations += 1
            if convergence or iterations >= self.max_iterations:
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
        self.k = 8
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

total_face_images = 451
total_digit_images = 5000

def extract_images(directory, filename, total_images=0):
    if total_images == 0:
		 total_images = total_face_images if directory == "facedata" else total_digit_images

    image_set = feature_extract.ImageSet(directory, filename)

    # Extract images, resize, and extract hog features
    print("File is %s/%s, whose images are %s by %s" % (image_set.directory, image_set.file_name, str(image_set.lenX), str(image_set.lenY), ))
    img_count = 0
    for img in image_set.extract_image():
        img.generate_hog_image()
        image_set.add_image(img)
        img_count += 1
        if img_count >= total_images:
            break

    # Vectorize each image and add to a new matrix M
    image_set.create_transpose_of_vectorized_images()

    # Transpose the matrix M
    return image_set.images_matrix

def import_labels(directory, filename, total_images=0):
    if total_images == 0:
		 total_images = total_face_images if directory == "facedata" else total_digit_images

    labels = []
    with open(directory + "/" + filename) as f:
        labels = f.readlines()
    return [x.strip() for x in labels][:total_images]

def main():
    if len(sys.argv) != 7:
        print("python classifier.py <image directory> <training data> <training labels> <test data> <test labels> <percent>")
        return 1

    max_training_images = 0
    if sys.argv[1] == "facedata":
        max_training_images = int(float(sys.argv[6])/100 * total_face_images)
    elif sys.argv[1] == "digitdata":
        max_training_images = int(float(sys.argv[6])/100 * total_digit_images)
    else:
    	print("Directory not supported")
    	print("Must be facedata or digitdata")
    	return

    print("Extracting training data")
    training_data = extract_images(sys.argv[1], sys.argv[2], total_images=max_training_images)
    training_labels = import_labels(sys.argv[1], sys.argv[3], total_images=max_training_images)

    assert len(training_data) == len(training_labels)

    print("Extracting testing data")
    test_data = extract_images(sys.argv[1], sys.argv[4])
    test_labels = import_labels(sys.argv[1], sys.argv[5])

    assert len(test_data) == len(test_labels)

    training_with_labels = []
    for i in range(0, len(training_data)):
        t = (training_labels[i], training_data[i])
        training_with_labels.append(t)

    # TODO: Train classifier here.
    print("Training bayes classifier")
    bayes = NaiveBayesClassifier(training_with_labels)
    print("Training perceptron classifier")
    perceptron = Perceptron(training_with_labels)
    rint("Training k nearest neighbors classifier")
    knn = KNearestNeighbors(training_with_labels)
    rint("Training random forest classifier")
    forest = RandomForestClassifier(training_with_labels)

    # TODO: Classify test data here.
    print("Classifying test data")
    bayes_correct = 0
    perceptron_correct = 0
    knn_correct = 0
    forest_correct = 0

    for i in range(0, len(test_data)):
        print("Classifying data point", i, "out of", len(test_data), end="\r")
        sys.stdout.flush()
        l = bayes.classifyData(test_data[i])
        if l == test_labels[i]:
            bayes_correct += 1
        l = perceptron.classifyData(test_data[i])
        if l == test_labels[i]:
            perceptron_correct += 1
        l = knn.classifyData(test_data[i])
        if l == test_labels[i]:
            knn_correct += 1
        l = forest.classifyData(test_data[i])
        if l == test_labels[i]:
            forest_correct += 1

    print("")
    print("Bayes:", float(bayes_correct)/len(test_data))
    print("Perceptron:", float(perceptron_correct)/len(test_data))
    print("KNN:", float(knn_correct)/len(test_data))
    print("Forest:", float(forest_correct)/len(test_data))


if __name__ == '__main__':
    main()
