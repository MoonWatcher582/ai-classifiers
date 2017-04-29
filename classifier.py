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
        pass

    def classifyData(self, data):
        pass


def main():
    # TODO: Feature extraction here
    # TODO: Train classifier here.
    # TODO: Classify test data here.
    pass

if __name__ == '__main__':
    main()
