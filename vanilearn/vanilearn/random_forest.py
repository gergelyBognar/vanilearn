import numpy as np

from decision_tree import DecisionTreeClassifier, most_common

class FeatureBagger:
    """
    Creates a random subset of feature indeces, for every node of the decision tree.
    """

    def __init__(self, feature_number, feature_bag_size):
        self._feature_number = feature_number
        self._feature_bag_size = feature_bag_size

    def feature_indeces(self):
        feature_indeces_list = np.random.choice(self._feature_number, self._feature_bag_size)
        for feature_index in feature_indeces_list:
            yield feature_index


class BaseRandomForest:
    def _get_base_estimator(self, **kwargs):
        raise NotImplementedError

    def __init__(self, categorical_feature_indeces=None, max_depth=None, criterion="gini", n_estimators=100, max_features="sqrt"):
        if categorical_feature_indeces is not None:
            self.categorical_feature_indeces = categorical_feature_indeces
        else:
            self.categorical_feature_indeces = list()
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.max_features = max_features

        # Initialize model list
        self._models = [self._get_base_estimator(
            categorical_feature_indeces=categorical_feature_indeces,
            max_depth=max_depth, criterion=criterion) for _ in range(n_estimators)]

    def _get_feature_bag_size(self, feature_number):
        if isinstance(self.max_features, int):
            return self.max_features
        if self.max_features == "sqrt":
            return max(int(np.floor(np.sqrt(feature_number))), 1)
        else:
            raise ValueError("max_features either has to be 'sqrt' or an integer number")

    def fit(self, features, labels):
        feature_number = features.shape[1]
        dataset_size = features.shape[0]

        feature_bag_size = self._get_feature_bag_size(feature_number)
        feature_bagger = FeatureBagger(feature_number, feature_bag_size)

        for model in self._models:
            # Bag data (sample with replacement)
            data_indeces = np.random.choice(dataset_size, dataset_size)
            features_sample = np.take(features, data_indeces, axis=0)
            labels_sample = np.take(labels, data_indeces)


            model.fit(features_sample, labels_sample, feature_bagger)

    def _summarize_predictions_mean(self, predictions_list):
        predictions = sum(predictions_list) / float(len(predictions_list))
        return predictions

    def _summarize_predictions(self, predictions_list, is_proba):
        raise NotImplementedError

    def _predict(self, features, is_proba):
        predictions_list = list()

        for model in self._models:

            if is_proba:
                predict_function = model.predict_proba
            else:
                predict_function = model.predict

            predictions_tree = predict_function(features)
            predictions_list.append(predictions_tree)

        predictions = self._summarize_predictions(predictions_list, is_proba)

        return predictions

    def predict(self, features):
        return self._predict(features, is_proba=False)

    def predict_proba(self, features):
        return self._predict(features, is_proba=True)

class RandomForestClassifier(BaseRandomForest):
    def _get_base_estimator(self, **kwargs):
        return DecisionTreeClassifier(**kwargs)

    def __init__(self, categorical_feature_indeces=None, max_depth=None, criterion="gini", n_estimators=100, max_features="sqrt"):
        super(RandomForestClassifier, self).__init__(
            categorical_feature_indeces=categorical_feature_indeces,
            max_depth=max_depth,
            criterion=criterion,
            n_estimators=n_estimators,
            max_features=max_features
        )

    def _summarize_predictions_majority(self, predictions_list):
        # Transpose, because list of arrays will be interpreted as matrix with shape (list_len, array_len),
        # but to iterate through the row of the predictions, shape (array_len, list_len) is needed.
        predictions_matrix = np.array(predictions_list).transpose()

        predictions = list()
        for predictions_datapoint in predictions_matrix:
            predictions.append(most_common(predictions_datapoint))

        return np.array(predictions)

    def _summarize_predictions(self, predictions_list, is_proba):
        if is_proba:
            return self._summarize_predictions_mean(predictions_list)
        else:
            return self._summarize_predictions_majority(predictions_list)
