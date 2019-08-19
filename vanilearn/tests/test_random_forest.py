import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifierSklearn
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from random_forest import RandomForestClassifier, FeatureBagger

from candidate_lifetime_dataset.data_loader_preprocesser import get_dataset


import unittest
from unittest import TestCase

class RandomForestTester(unittest.TestCase):

    #-------------------------------------------------------------------------
    # Feature bagger
    #-------------------------------------------------------------------------

    def test_feature_bagger(self):
        feature_bagger = FeatureBagger(feature_number=5, feature_bag_size=2)

        for _ in range(10):
            print("")
            for feature_index in feature_bagger.feature_indeces():
                print(feature_index)



    def test_random_foerst_fit_predict(self):
        model = RandomForestClassifier(n_estimators=100)

        features = np.array([
            [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1],
            [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1],
            [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1],
            [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1],
            [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1],
            [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1],
        ])
        labels = np.array([
            0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        ])

        model.fit(features, labels)

        """
        for tree in model._models:
            print("=================================")
            from pprint import pprint
            pprint(tree._node)

        for tree in model._models:
            print(tree.predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])))
        """

        predictions = model.predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))

        self.assertEqual(predictions.tolist(), [0, 1, 1, 0])

    def _compare_sklearn_model(self, features, labels, categorical_features, feature_names, model_config=None):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        # Encode categorical features
        for categorical_feature in categorical_features:
            features[categorical_feature] = pd.factorize(features[categorical_feature])[0]

        # Find categorical feature indeces
        categorical_feature_indeces = [feature_names.index(categorical_feature) for categorical_feature in categorical_features]

        imputer = Imputer()
        features = imputer.fit_transform(features)

        features_training, features_test, labels_training, labels_test = train_test_split(features, labels, test_size=0.3)


        model_vanilearn = RandomForestClassifier(categorical_feature_indeces=categorical_feature_indeces, **model_config)
        model_sklearn = RandomForestClassifierSklearn(**model_config)

        for model in [model_sklearn, model_vanilearn]:

            model.fit(features_training, labels_training)

            predictions_training = model.predict(features_training)
            predictions_test = model.predict(features_test)

            accuracy_training = accuracy_score(labels_training, predictions_training)
            accuracy_test = accuracy_score(labels_test, predictions_test)

            print("")
            print(model)
            print(accuracy_training)
            print(accuracy_test)

    """
    def test_sklearn_titanic(self):
        df = pd.read_csv("titanic_dataset/train.csv")

        feature_names = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]

        features = df[feature_names]
        labels = df["Survived"].values

        categorical_features = ["Sex", "Cabin", "Embarked"]
        categorical_feature_indeces = [1, 6, 7]
        #categorical_feature_indeces = None



        self._compare_sklearn_model(features, labels, categorical_features, feature_names)
    """

    def test_31_credit_g(self):
        df = pd.read_csv("dataset_31_credit-g/dataset_31_credit-g.csv")

        label_name = "class"

        labels = df[label_name]

        feature_names = list(df.columns)
        feature_names.remove(label_name)

        features = df[feature_names]

        categorical_features = [
            "checking_status",
            "credit_history",
            "purpose",
            "savings_status",
            "employment",
            "personal_status",
            "other_parties",
            "property_magnitude",
            "other_payment_plans",
            "housing",
            "job",
            "own_telephone",
            "foreign_worker"
        ]

        model_config = {
            "max_depth": 8,
            "n_estimators": 50
        }

        self._compare_sklearn_model(features, labels, categorical_features, feature_names, model_config)


    """
    def test_sklearn_candidate_lifetime(self):
        df = get_dataset(
            min_date = '2017-01-01',
            max_date = '2018-01-01',
            data_file='candidate_lifetime_dataset/assignment_cand.csv'
        )

        feature_names = [
            'job_position', 'branche_name', 'company_province', 'salary_per_hour',
            'candidate_province', 'edu_type', 'gender', 'age_start_assignment_int',
            'assignment_start_mont', 'is_same_provance',
        ]

        features = df[feature_names]
    """









if __name__ == '__main__':
    unittest.main()
