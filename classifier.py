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
        self.labels = {}

        # This is a list of maps. Each entry in the list for the given feature
        # and the map gives the probabilty of the feature's value given the
        # different values of the label.
        self.feature_probabilties = {}

        # This is in the same format as above, but is the counts instead of
        # probabilities.
        self.feature_counts = {}

        for feat_num in {0 ... len(data_with_labels[0][1]) - 1}:
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
                prod_map[label] = dict()
                total_count = 0
                for value, count in value_map.iteritems():
                    total_count += count
                for value, count in value_map.iteritems():
                    probability = float(count)/total_count
                    prob_map[label][value] = probability
                    if probability / 2 < self.min_probability:
                        self.min_probability = probability / 2


    def classifyData(self, data):
        probabilties = {}
        for label in self.labels:
            # Calculate the probabilty of each label given the observed
            # features.
            label_prob = 1
            for feature in data:
                feature_prob =
                    self.feature_probabilties.get(label).get(feature)
                if feature_prob == None:
                    feature_prob = self.min_probability
                label_prob *= feature_prob
            probabilities.append(label_prob)

        # Return the label with the maximum probability.
        return self.labels[probabilities.index(max(probabilities))]


def main():
    # TODO: Feature extraction here
    # TODO: Train classifier here.
    # TODO: Classify test data here.
    pass

if __name__ == '__main__':
    main()
