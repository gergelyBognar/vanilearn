# vanilearn
An implementation of decision tree and random forest algorithms. The purpose of this repo is understanding these algorithms, and extend them to better handle categorical variables.

## Usage
```python
import numpy as np
from vanilearn.decision_tree import DecisionTreeClassifier

features = np.array([
    [0.1, 0],
    [0.7, 2],
    [0.2, 1],
    [0.3, 1],
    [0.2, 1],
    [1.1, 0],
    [1.5, 2],
    [0.9, 0],
    [0.4, 1],
    [1.2, 1],
    [1.0, 1],
])
labels = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])

model = DecisionTreeClassifier(categorical_feature_indeces=[1])

model.fit(features, labels)

model.to_csv("./model_interpretation.csv")
```
