import operator
import random

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


def main():
    # TODO: Feature extraction here
    # TODO: Train classifier here.
    # TODO: Classify test data here.
    data_with_labels = [
            (0, [0,0]),
            (1, [1,1]),
            (0, [1,1]),
            (0, [0,2]),
    ]
    c = NaiveBayesClassifier(data_with_labels)
    print c.classifyData([0,2])

    # Simple random forest test.
    c = RandomForestClassifier(None)
    for i in range(0, 10):
        print c.classifyData([0, 0])

if __name__ == '__main__':
    main()
