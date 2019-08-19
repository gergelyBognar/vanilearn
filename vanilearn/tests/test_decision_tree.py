import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifierSklearn

from decision_tree import NumericalSplitter, ExhaustiveCategoricalSplitter, \
    SmartClassifierCategoricalSplitter, SmartRegressorCategoricalSplitter, \
    DecisionTreeClassifier

import unittest
from unittest import TestCase

class DecisionTreeTester(unittest.TestCase):

    #-------------------------------------------------------------------------
    # Numerical splitter
    #-------------------------------------------------------------------------

    def test_numerical_splitter_all_splits(self):
        splitter = NumericalSplitter()

        feature = np.array([4, 3, 3, 1, 4, 2, 3, 1, 1, 1, 4,])
        labels = np.array([14, 13, 13, 11, 14, 12, 13, 11, 11, 11, 14,])

        splits = list()
        for split in splitter.all_splits(feature, labels):
            splits.append(split)

        self.assertEqual(sorted(splits), [1, 2, 3, 4])

    def test_numerical_splitter_split_labels(self):
        splitter = NumericalSplitter()

        feature = np.array([4, 3, 3, 1, 4, 2, 3, 1, 1, 1, 4,])
        labels = np.array([14, 13, 13, 11, 14, 12, 13, 11, 11, 11, 14,])

        split = 2
        labels_positive, labels_negative = splitter.split_labels(split, feature, labels)

        self.assertEqual(labels_positive.tolist(), [14, 13, 13, 14, 13, 14])
        self.assertEqual(labels_negative.tolist(), [11, 12, 11, 11, 11])

    def test_numerical_splitter_split_dataset(self):
        splitter = NumericalSplitter()

        features = np.array([[1, 4], [2, 3], [1, 3], [4, 1], [4, 4], [1, 2], [2, 3], [1, 1], [2, 1], [3, 1], [4, 4]])
        labels = np.array([14, 13, 13, 11, 14, 12, 13, 11, 11, 11, 14,])

        split = 2
        feature_index = 1
        features_positive, labels_positive, features_negative, labels_negative = \
                splitter.split_dataset(split, features[:,feature_index], features, labels)

        self.assertEqual(features_positive.tolist(), [[1, 4], [2, 3], [1, 3], [4, 4], [2, 3], [4, 4]])
        self.assertEqual(features_negative.tolist(), [[4, 1], [1, 2], [1, 1], [2, 1], [3, 1]])
        self.assertEqual(labels_positive.tolist(), [14, 13, 13, 14, 13, 14])
        self.assertEqual(labels_negative.tolist(), [11, 12, 11, 11, 11])

    #-------------------------------------------------------------------------
    # Exhaustive categorical splitter
    #-------------------------------------------------------------------------

    def test_exhaustive_categorical_splitter_all_splits(self):
        splitter = ExhaustiveCategoricalSplitter()

        feature = np.array([4, 3, 3, 1, 4, 2, 3, 1, 1, 1, 4,])
        labels = np.array([14, 13, 13, 11, 14, 12, 13, 11, 11, 11, 14,])

        all_splits = list()
        for split in splitter.all_splits(feature, labels):
            all_splits.append(split)

        self.assertEqual(all_splits, [[1], [2], [3], [4], [1, 2], [1, 3], [1, 4]])

    def test_exhaustive_categorical_splitter_split_labels(self):
        splitter = ExhaustiveCategoricalSplitter()

        feature = np.array([4, 3, 3, 1, 4, 2, 3, 1, 1, 1, 4,])
        labels = np.array([14, 13, 13, 11, 14, 12, 13, 11, 11, 11, 14,])

        split = [3, 4]
        labels_positive, labels_negative = splitter.split_labels(split, feature, labels)

        self.assertEqual(labels_positive.tolist(), [14, 13, 13, 14, 13, 14])
        self.assertEqual(labels_negative.tolist(), [11, 12, 11, 11, 11])

    def test_exhaustive_categorical_splitter_split_dataset(self):
        splitter = ExhaustiveCategoricalSplitter()

        features = np.array([[1, 4], [2, 3], [1, 3], [4, 1], [4, 4], [1, 2], [2, 3], [1, 1], [2, 1], [3, 1], [4, 4]])
        labels = np.array([14, 13, 13, 11, 14, 12, 13, 11, 11, 11, 14,])

        split = [3, 4]
        feature_index = 1
        features_positive, labels_positive, features_negative, labels_negative = \
                splitter.split_dataset(split, features[:,feature_index], features, labels)

        self.assertEqual(features_positive.tolist(), [[1, 4], [2, 3], [1, 3], [4, 4], [2, 3], [4, 4]])
        self.assertEqual(features_negative.tolist(), [[4, 1], [1, 2], [1, 1], [2, 1], [3, 1]])
        self.assertEqual(labels_positive.tolist(), [14, 13, 13, 14, 13, 14])
        self.assertEqual(labels_negative.tolist(), [11, 12, 11, 11, 11])

    #-------------------------------------------------------------------------
    # Smart classifier categorical splitter
    #-------------------------------------------------------------------------

    def test_smart_classifer_categorical_splitter_all_splits_error(self):
        splitter = SmartClassifierCategoricalSplitter()

        feature = np.array([1, 2, 3, 4, 5, 6])
        labels = np.array([0, 1, 0, 2, 1, 0])

        with self.assertRaises(ValueError) as context:
            split = next(splitter.all_splits(feature, labels))

        self.assertEqual("This is not a binary classification problem, smart splitter can't be used.", str(context.exception))

    def test_smart_classifer_categorical_splitter_all_splits(self):
        splitter = SmartClassifierCategoricalSplitter()

        # order of feature categories, from lowest to highest class probalility of class 0: 2, 1, 3
        feature = np.array([1, 2, 1, 2, 3, 3, 1, 1, 2, 1, 3, 2, 1, 2, 2, 3, 3, 3])
        labels = np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0])

        all_splits = list()
        for split in splitter.all_splits(feature, labels):
            all_splits.append(split)

        self.assertEqual(all_splits, [[2], [2, 1]])

    def test_smart_classifer_categorical_splitter_split_labels(self):
        splitter = SmartClassifierCategoricalSplitter()

        feature = np.array([4, 3, 3, 1, 4, 2, 3, 1, 1, 1, 4,])
        labels = np.array([14, 13, 13, 11, 14, 12, 13, 11, 11, 11, 14,])

        split = [3, 4]
        labels_positive, labels_negative = splitter.split_labels(split, feature, labels)

        self.assertEqual(labels_positive.tolist(), [14, 13, 13, 14, 13, 14])
        self.assertEqual(labels_negative.tolist(), [11, 12, 11, 11, 11])

    def test_smart_classifer_categorical_splitter_split_dataset(self):
        splitter = SmartClassifierCategoricalSplitter()

        features = np.array([[1, 4], [2, 3], [1, 3], [4, 1], [4, 4], [1, 2], [2, 3], [1, 1], [2, 1], [3, 1], [4, 4]])
        labels = np.array([14, 13, 13, 11, 14, 12, 13, 11, 11, 11, 14,])

        split = [3, 4]
        feature_index = 1
        features_positive, labels_positive, features_negative, labels_negative = \
                splitter.split_dataset(split, features[:,feature_index], features, labels)

        self.assertEqual(features_positive.tolist(), [[1, 4], [2, 3], [1, 3], [4, 4], [2, 3], [4, 4]])
        self.assertEqual(features_negative.tolist(), [[4, 1], [1, 2], [1, 1], [2, 1], [3, 1]])
        self.assertEqual(labels_positive.tolist(), [14, 13, 13, 14, 13, 14])
        self.assertEqual(labels_negative.tolist(), [11, 12, 11, 11, 11])


    #-------------------------------------------------------------------------
    # Smart regressor categorical splitter
    #-------------------------------------------------------------------------

    def test_smart_regressor_categorical_splitter_all_splits(self):
        splitter = SmartRegressorCategoricalSplitter()

        # order of feature categories, from lowest to highest class probalility of class 0: 2, 1, 3
        feature = np.array([1, 2, 1, 2, 3, 3, 1, 1, 2, 1, 3, 2, 1, 2, 2, 3, 3, 3])
        labels = np.array([6.2, 4.1, 6.1, 3.8, 8.1, 8.2, 5.8, 5.9, 4.2, 6.4, 7.9, 5.1, 5.5, 3.5, 3.6, 8.0, 7.6, 7.8])

        all_splits = list()
        for split in splitter.all_splits(feature, labels):
            all_splits.append(split)

        self.assertEqual(all_splits, [[2], [2, 1]])

    def test_smart_regressor_categorical_splitter_split_labels(self):
        splitter = SmartRegressorCategoricalSplitter()

        feature = np.array([4, 3, 3, 1, 4, 2, 3, 1, 1, 1, 4,])
        labels = np.array([14, 13, 13, 11, 14, 12, 13, 11, 11, 11, 14,])

        split = [3, 4]
        labels_positive, labels_negative = splitter.split_labels(split, feature, labels)

        self.assertEqual(labels_positive.tolist(), [14, 13, 13, 14, 13, 14])
        self.assertEqual(labels_negative.tolist(), [11, 12, 11, 11, 11])

    def test_smart_regressor_categorical_splitter_split_dataset(self):
        splitter = SmartRegressorCategoricalSplitter()

        features = np.array([[1, 4], [2, 3], [1, 3], [4, 1], [4, 4], [1, 2], [2, 3], [1, 1], [2, 1], [3, 1], [4, 4]])
        labels = np.array([14, 13, 13, 11, 14, 12, 13, 11, 11, 11, 14,])

        split = [3, 4]
        feature_index = 1
        features_positive, labels_positive, features_negative, labels_negative = \
                splitter.split_dataset(split, features[:,feature_index], features, labels)

        self.assertEqual(features_positive.tolist(), [[1, 4], [2, 3], [1, 3], [4, 4], [2, 3], [4, 4]])
        self.assertEqual(features_negative.tolist(), [[4, 1], [1, 2], [1, 1], [2, 1], [3, 1]])
        self.assertEqual(labels_positive.tolist(), [14, 13, 13, 14, 13, 14])
        self.assertEqual(labels_negative.tolist(), [11, 12, 11, 11, 11])

    #-------------------------------------------------------------------------
    # Decision tree classifier fit
    #-------------------------------------------------------------------------

    def test_decision_tree_classifier_fit(self):
        model = DecisionTreeClassifier()

        # XOR problem
        features = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1]])
        labels = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])

        model.fit(features, labels)

        """
        from pprint import pprint
        pprint(model._node)
        """

        predictions = model.predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))

        self.assertEqual(predictions.tolist(), [0, 1, 1, 0])

    def test_decision_tree_classifier_numerical_split(self):
        model = DecisionTreeClassifier()

        # feature 3, 1 -> label 1
        # feature 2, 0 -> label 0
        features = np.array([[3, 0], [3, 0], [3, 0], [2, 0], [2, 0], [2, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 0]])
        labels = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0])

        model.fit(features, labels)

        """
        from pprint import pprint
        pprint(model._node)
        """

    def test_decision_tree_classifier_numerical_split_hard(self):
        model = DecisionTreeClassifier()

        # feature 3, 1 -> label 1
        # feature 2, 0 -> label 0
        features = np.array([[0, 0], [0, 0], [1, 0], [1, 0], [2, 0], [2, 0], [3, 0], [3, 0], [4, 0], [4, 0], [5, 0], [5, 0], \
            [6, 0], [6, 0], [7, 0], [7, 0], [8, 0], [8, 0], [9, 0], [9, 0], [10, 0], [10, 0], [11, 0], [11, 0]])
        labels = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, ])


        model.fit(features, labels)

        """
        print("test_decision_tree_classifier_numerical_split_hard")
        from pprint import pprint
        pprint(model._node)
        """

        predictions = model.predict(np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0]]))

        self.assertEqual(predictions.tolist(), [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])


    def test_decision_tree_classifier_smart_categorical_split(self):
        model = DecisionTreeClassifier(categorical_feature_indeces=[0,1])

        # feature[0] 3, 1 -> label 1
        # feature[0] 2, 0 -> label 0
        features = np.array([[3, 0], [3, 0], [3, 0], [2, 0], [2, 0], [2, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 0]])
        labels = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0])

        model.fit(features, labels)

        """
        from pprint import pprint
        pprint(model._node)

        print(model._categorical_splitter)
        """


    def test_decision_tree_classifier_smart_categorical_split_hard(self):
        model = DecisionTreeClassifier(categorical_feature_indeces=[0,1])

        # feature 3, 1 -> label 1
        # feature 2, 0 -> label 0
        features = np.array([[0, 0], [0, 0], [1, 0], [1, 0], [2, 0], [2, 0], [3, 0], [3, 0], [4, 0], [4, 0], [5, 0], [5, 0], \
            [6, 0], [6, 0], [7, 0], [7, 0], [8, 0], [8, 0], [9, 0], [9, 0], [10, 0], [10, 0], [11, 0], [11, 0]])
        labels = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, ])

        model.fit(features, labels)

        print("test_decision_tree_classifier_smart_categorical_split_hard")
        from pprint import pprint
        pprint(model._node)

        predictions = model.predict(np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0]]))

        self.assertEqual(predictions.tolist(), [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])


    def test_decision_tree_classifier_exhaustive_categorical_split(self):
        model = DecisionTreeClassifier(categorical_feature_indeces=[0,1])

        # feature[0] 3, 1 -> label 1
        # feature[0] 2, 0 -> label 0, 2
        features = np.array([[3, 0], [3, 0], [3, 0], [2, 0], [2, 0], [2, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 1]])
        labels = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 2])

        model.fit(features, labels)

        """
        from pprint import pprint
        pprint(model._node)

        print(model._categorical_splitter)
        """

    #-------------------------------------------------------------------------
    # Validate with sklearn
    #-------------------------------------------------------------------------

    def _compare_sklearn_dataset(self, dataset):
        dataset = load_iris()

        features = dataset.data
        labels = dataset.target

        model_sklearn = DecisionTreeClassifierSklearn()

        model_sklearn.fit(features, labels)
        predictions_sklearn = model_sklearn.predict(features)


        model = DecisionTreeClassifier()

        model.fit(features, labels)
        predictions = model.predict(features)

        self.assertEqual(predictions.tolist(), predictions_sklearn.tolist())

    def test_sklearn_iris(self):
        dataset = load_iris()
        self._compare_sklearn_dataset(dataset)




if __name__ == '__main__':
    unittest.main()
