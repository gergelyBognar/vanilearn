from itertools import combinations
from collections import Counter
import csv

import numpy as np

"""
TODO: node keys (like "is_terminal") to GLOBAL_CONSTANTS?
"""


def most_common(array):
    counts = Counter(array)
    return counts.most_common(1)[0][0]


class BaseSplitter:
    def all_splits(self, feature, labels):
        raise NotImplementedError

    def _get_split_mask(self, split, feature):
        raise NotImplementedError

    def split_labels(self, split, feature, labels):
        split_mask = self._get_split_mask(split, feature)

        labels_positive = labels[split_mask]
        labels_negative = labels[~split_mask]

        return labels_positive, labels_negative

    def split_dataset(self, split, feature, features, labels):
        split_mask = self._get_split_mask(split, feature)

        features_positive = features[split_mask,:]
        features_negative = features[~split_mask,:]

        labels_positive = labels[split_mask]
        labels_negative = labels[~split_mask]

        return features_positive, labels_positive, features_negative, labels_negative

    def decide(self, split, value):
        raise NotImplementedError


class NumericalSplitter(BaseSplitter):
    def __init__(self):
        super(NumericalSplitter, self).__init__()

    def all_splits(self, feature, labels):
        """
        Iterates through all unique values of the feature, except the greatest one.

        Why except the greatest one: because the split rule is feature > split_value,
        if the greatest number is returned as split_value, then feature > split_value would be all false,
        and the positive branch of the split would be empty. This is not desirable.
        """
        unique_values = np.unique(feature) # np.unique() returns the ordered unique values
        for split_value in unique_values[:-1]:
            yield split_value

    def _get_split_mask(self, split_value, feature):
        """ Get boolean mask, where feature is greater than the split value. """
        split_mask = feature > split_value
        return split_mask

    def decide(self, split_value, value):
        return value > split_value


class BaseCategoricalSplitter(BaseSplitter):
    def __init__(self):
        super(BaseCategoricalSplitter, self).__init__()

    def _get_split_mask(self, split_list, feature):
        split_mask = np.isin(feature, split_list)
        return split_mask

    def decide(self, split_list, value):
        return value in split_list


class ExhaustiveCategoricalSplitter(BaseCategoricalSplitter):
    """ Loops through all possible 2^n / 2 - 1 category splits to two. """

    def __init__(self):
        super(ExhaustiveCategoricalSplitter, self).__init__()

    def all_splits(self, feature, labels):
        """
        Generates all possible ways j (=i+1) element can be selected, form categories,
        wher j ranges from 1 to len(categories)-1.
        Then loop through all these combinations, if the complementer of the combination
        was not yielded before, then yield it, otherwise omit it.

        Yields:
            combination (list): a list of categories, that represent a split in the following way:
                the datapoint, where the feature to split on is in combination, then it belongs
                to branch positive, otherwise belongs to branch negative.
        """

        categories = np.unique(feature)

        all_category_combinations = [list(combination) for i in range(len(categories)-1) \
            for combination in combinations(categories, i+1)]

        combinations_yielded = list()
        for combination in all_category_combinations:
            complementer_categories = [category for category in categories if category not in combination]
            if complementer_categories not in combinations_yielded:
                combinations_yielded.append(combination)
                yield combination


class BaseSmartCategoricalSplitter(BaseCategoricalSplitter):
    def __init__(self):
        super(BaseSmartCategoricalSplitter, self).__init__()

    def _argsort_categories(self, categories, feature, labels):
        raise NotImplementedError

    def all_splits(self, feature, labels):
        categories = np.unique(feature)

        ordered_category_indeces = self._argsort_categories(categories, feature, labels)

        for index in range(len(ordered_category_indeces) - 1):
            # The first list the yield generates should be
            # the first category in the ordered list, hence the range(len() - 1) .
            # The last list the yield generates should be
            # all but the last category in the ordered list, hence the [:index+1].
            split_list = categories[ordered_category_indeces[:index+1]].tolist()
            yield split_list


class SmartClassifierCategoricalSplitter(BaseSmartCategoricalSplitter):
    """
    In case of binary classification this smart algorith can find the optimal split,
    without having to look at all combination of the categories.

    Based on: https://fr.mathworks.com/help/stats/splitting-categorical-predictors-for-multiclass-classification.html
    """

    def __init__(self):
        super(SmartClassifierCategoricalSplitter, self).__init__()

    def _argsort_categories(self, categories, feature, labels):
        """ pi = E(y=y_selected | x=xi) """

        classes = np.unique(labels)

        # Check if binary classification problem.
        if len(classes) > 2:
            raise ValueError("This is not a binary classification problem, smart splitter can't be used.")

        class_selected = classes[0] # Select one of the classes

        class_probabilies = list()
        for category in categories:
            category_mask = feature == category

            labels_at_category = labels[category_mask]

            class_probability = (labels_at_category == class_selected).sum() / float(len(labels_at_category))
            class_probabilies.append(class_probability)

        ordered_category_indeces = np.argsort(np.array(class_probabilies))

        return ordered_category_indeces


class SmartRegressorCategoricalSplitter(BaseSmartCategoricalSplitter):
    """
    In case of regression this smart algorith can find the optimal split,
    without having to look at all combination of the categories.

    Based on: https://fr.mathworks.com/help/stats/splitting-categorical-predictors-for-multiclass-classification.html
    Also based on: https://en.wikipedia.org/wiki/Mean_and_predicted_response
    """

    def __init__(self):
        super(SmartRegressorCategoricalSplitter, self).__init__()

    def _argsort_categories(self, categories, feature, dependent):
        """ y_mr_i = E(y | x=xi) """

        mean_responses = list()
        for category in categories:
            category_mask = feature == category

            dependent_at_category = dependent[category_mask]
            mean_response = dependent_at_category.mean()
            mean_responses.append(mean_response)

        ordered_category_indeces = np.argsort(np.array(mean_responses))

        return ordered_category_indeces


class TableFormater():
    def __init__(self):
        self._lines = []

    def add_rule(self, node, rules=None):
        if rules is None:
            rules = ()

        if node["is_terminal"]:
            rules_with_terminal = rules + ("prediction: {}, \nprobabilities: {}, \nsample_count:{}".format(
                node["prediction"], node["probabilities"], node["sample_count"]
            ),)
            self._lines.append(rules_with_terminal)
        else:
            rules_with_new = \
                rules \
                + ("feature_index: {}, \nsplit: {}".format(node["feature_index"], node["split"]),) \
                + ("\nprobabilities: {}, \nsample_count:{}".format(node["probabilities"], node["sample_count"]),)
            self.add_rule(node["node_positive"], rules_with_new + ("node_positive",))
            self.add_rule(node["node_negative"], rules_with_new + ("node_negative",))

class BaseDecisionTree:

    def __init__(self, categorical_feature_indeces=None, max_depth=None, min_samples_leaf=None):
        if categorical_feature_indeces is not None:
            self.categorical_feature_indeces = categorical_feature_indeces
        else:
            self.categorical_feature_indeces = list()
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        # Init basic splitters
        self._numerical_splitter = NumericalSplitter()
        self._categorical_splitter = ExhaustiveCategoricalSplitter()

        # fit function will create this
        self._node = None


    def _calc_criterion(self, labels_positive, labels_negative):
        raise NotImplementedError

    def _calculate_prediction(self, labels):
        raise NotImplementedError

    def _calculate_probabilities(self, labels):
        raise NotImplementedError

    def _get_splitter(self, feature_index):
        """ Decide if categorical or numerical splitters and splits """
        if feature_index in self.categorical_feature_indeces:
            splitter = self._categorical_splitter
        else:
            splitter = self._numerical_splitter

        return splitter

    def _is_variance(self, array):
        return len(np.unique(array)) > 1

    def _create_terminal_node(self, labels):
        """ Calculate predicted class or value, and probabilities for current node. """
        node = {
            "is_terminal": True,
            "prediction": self._calculate_prediction(labels),
            "probabilities": self._calculate_probabilities(labels),
            "sample_count": len(labels),
        }

        return node

    def _grow_node(self, features, labels, depth, feature_bagger=None):
        """
        Args:
            features (numpy array): with a shape of (dataset_size, feature_numer)
            labels (numpy array): with a shape of (dataset_size)
            depth (integer): the depth of the node being calculated
            feature_bagger (object with function feature_indeces): optional, function feature_indeces generates
                feature indeces to consider at the current node
        """

        stop_criterion = False
        # If this leaf is pure, then stop
        if not self._is_variance(labels):
            stop_criterion = True
        # Stop criterion based on depth
        elif self.max_depth is not None:
            if depth >= self.max_depth:
                stop_criterion = True

        if stop_criterion:
            # Set current node as terminal node
            node = self._create_terminal_node(labels)
            return node

        else:
            # In the case of feature bagging, feature_bagger defines the features
            # to consider at this node. Otherwise all features are used.
            if feature_bagger is not None:
                feature_indeces = feature_bagger.feature_indeces()
            else:
                feature_indeces = range(features.shape[1])

            best_score = None
            best_feature_index = None
            best_split = None


            # Iterate through features
            for feature_index in feature_indeces:

                # Get current feature
                feature = features[:,feature_index]

                # Decide if categorical or numerical splitter
                splitter = self._get_splitter(feature_index)

                # Iterate through all splits
                for split in splitter.all_splits(feature, labels):
                    # Split labels, based on split rule and feature
                    labels_positive, labels_negative = splitter.split_labels(split, feature, labels)

                    # Calculate score
                    score = self._calc_criterion(labels_positive, labels_negative)

                    # If the new score is better (smaller) than save score, feature index and split
                    if (best_score is None) or (score < best_score):
                        if (self.min_samples_leaf is None) or ((len(labels_positive) >= self.min_samples_leaf) and (len(labels_negative) >= self.min_samples_leaf)):
                            best_score = score
                            best_feature_index = feature_index
                            best_split = split

            stop_criterion_after_search = False

            if (best_score is None) or (best_feature_index is None) or (best_split is None):
                # If these variables are None, then there was no valid split in feature
                stop_criterion_after_search = True
            """
            elif not self._is_variance(features[:,best_feature_index]):
                # If there is no variation in the best feature (all the values are the same),
                # then stop and convert this node to terminal node.
                # TODO: other possible solution is to further search in features,
                # because there might be a better feature, we only selected the wrong one,
                # because of the feature bagger in random forest
                stop_criterion_after_search = True
            """

            if stop_criterion_after_search:
                # Set current node as terminal node
                node = self._create_terminal_node(labels)
                return node


            # Decide if categorical or numerical splitter
            best_splitter = self._get_splitter(best_feature_index)

            features_positive, labels_positive, features_negative, labels_negative = \
                best_splitter.split_dataset(best_split, features[:,best_feature_index], features, labels)


            # TODO: thi is not necessary, if this error doesn't occur anymore
            if len(labels_positive) == 0 or len(labels_negative) == 0:
                import pdb; pdb.set_trace()
                raise ValueError("Hoppa I got you: len(labels_positive) == 0 or len(labels_negative) == 0")


            # Save everything to the current node
            node = {
                "is_terminal": False,
                "score": best_score,
                "feature_index": best_feature_index,
                "split": best_split,
                "node_positive": self._grow_node(features_positive, labels_positive, depth+1, feature_bagger),
                "node_negative": self._grow_node(features_negative, labels_negative, depth+1, feature_bagger),
                "probabilities": self._calculate_probabilities(labels),
                "sample_count": len(labels),
            }

            return node

    def _fit(self, features, labels, feature_bagger=None):

        depth = 0
        self._node = self._grow_node(features, labels, depth, feature_bagger)

    def _predict_datapoint(self, node, datapoint, predict_type):
        # If terminal node, return prediction
        if node["is_terminal"]:
            return node[predict_type]

        # If not terminal node, make a decision, based on the saved split and feature index
        # and return whatever the resulting branch returns.
        else:
            # Decide if categorical or numerical splitters and splits
            splitter = self._get_splitter(node["feature_index"])

            # Make decision based on the datapoint, saved split and feature index
            decision = splitter.decide(node["split"], datapoint[node["feature_index"]])

            if decision:
                node_selected = node["node_positive"]
            else:
                node_selected = node["node_negative"]

            # Recursively go deeper in the tree
            return self._predict_datapoint(node_selected, datapoint, predict_type)

    def _predict(self, features, predict_type):
        predictions_list = list()

        for datapoint in features:
            predictions_list.append(self._predict_datapoint(self._node, datapoint, predict_type))

        predictions = np.array(predictions_list)

        return predictions

    def to_csv(self, file_path):
        table_formater = TableFormater()
        table_formater.add_rule(self._node)
        with open(file_path, "w") as filehandler:
            csv_writer = csv.writer(filehandler, delimiter=',')
            for line in table_formater._lines:
                csv_writer.writerow(line)



class DecisionTreeClassifier(BaseDecisionTree):
    """
    categorical_feature_indeces (sequence or numpy array)
    criterion (str) either "gini" or "entropy"
    """

    def __init__(self, categorical_feature_indeces=None, max_depth=None, min_samples_leaf=None, criterion="gini"):
        super(DecisionTreeClassifier, self).__init__(
            categorical_feature_indeces=categorical_feature_indeces,
            max_depth=max_depth, min_samples_leaf=min_samples_leaf
        )

        self.criterion = criterion

        # These will be populated in fit
        self._classes = None

    def _calc_criterion(self, labels_positive, labels_negative):
        if self.criterion == "gini":
            return self._gini_criterion(labels_positive, labels_negative)
        elif self.criterion == "entropy":
            return self._entropy_criterion(labels_positive, labels_negative)
        else:
            ValueError("criterion shall either be 'gini' or 'entropy'")



    def _gini_criterion(self, labels_positive, labels_negative):
        """
        Args:
            labels_positive (1D numpy array): labels when the split condition is true,
                in other words one branch of the split
            labels_negative (1D numpy array): labels when the split condition is false,
                in other words the other branch of the split

        Returns:
            gini_index (float): the smaller the better, positive number or zero
        """
        # Calculate gini index for one half of split
        def gini_index_leaf(labels):
            classes = np.unique(labels)

            probability_square_sum = 0
            for class_ in classes:
                probability = (labels == class_).mean()
                probability_square_sum += probability ** 2

            return 1 - probability_square_sum

        # Calculate gini inddex, the weighted sum of the gini indeces of the two halfs of the split
        dataset_size = len(labels_positive) + len(labels_negative)
        gini_index = \
            len(labels_positive) / float(dataset_size) * gini_index_leaf(labels_positive) + \
            len(labels_negative) / float(dataset_size) * gini_index_leaf(labels_negative)

        return gini_index

    def _entropy_criterion(self, labels_positive, labels_negative):
        pass

    def _calculate_prediction(self, labels):
        most_common_class = most_common(labels)

        return most_common_class

    def _calculate_probabilities(self, labels):
        probabilities = list()
        for class_ in self._classes:
            probability = (labels == class_).mean()
            probabilities.append(probability)
        return probabilities

    def fit(self, features, labels, feature_bagger=None):
        self._classes = np.unique(labels) # np.unique also sorts values

        # If binary classifier, then smart splitter can be used
        if len(self._classes) <= 2:
            self._categorical_splitter = SmartClassifierCategoricalSplitter()

        self._fit(features, labels, feature_bagger)

    def predict(self, features):
        predictions = self._predict(features, predict_type="prediction")
        return predictions

    def predict_proba(self, features):
        """
        Returns:
            probabilities (numpy array): with the shape (datapoint_size, cluster_number)
            each column corresponds to a class in the self.classes_ respectively. self.classes_
            is the unique sorted values of the training labels, when the model was fit.
        """
        probabilities = self._predict(features, predict_type="probabilities")
        return probabilities

    def predict_proba(self, features):
        self._predict(features, predict_type="probabilities")


class DecisionTreeRegressor(BaseDecisionTree):
    """
    categorical_feature_indeces (sequence or numpy array)
    """

    def __init__(self, categorical_feature_indeces=None, max_depth=None, min_samples_leaf=None):
        super(DecisionTreeRegressor, self).__init__(
            categorical_feature_indeces=categorical_feature_indeces,
            max_depth=max_depth, min_samples_leaf=min_samples_leaf
        )

        self._categorical_splitter = SmartRegressorCategoricalSplitter()


    def _calculate_prediction(self, labels):
        """ Mean value of dependent """
        return labels.mean()


    def _calc_criterion(self, labels_positive, labels_negative):
        """ Mean Squared Error """
        prediction_positive = self._calculate_prediction(labels_positive)
        prediction_negative = self._calculate_prediction(labels_negative)
        errors = np.concatenate([(labels_positive - prediction_positive), (labels_negative - prediction_negative)])
        mean_squared_error = (np.square(errors)).mean()
        return mean_squared_error


    def _calculate_probabilities(self, labels):
        return None


    def fit(self, features, labels, feature_bagger=None):
        self._fit(features, labels, feature_bagger)


    def predict(self, features):
        predictions = self._predict(features, predict_type="prediction")
        return predictions
