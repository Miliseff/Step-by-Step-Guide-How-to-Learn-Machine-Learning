# Random Forests

**Learning Objectives**

1. Understand the concept of ensembling and bagging  
2. Build a RandomForestClassifier model with scikit-learn  
3. Tune key hyperparameters: `n_estimators`, `max_depth`, `max_features`  
4. Evaluate classification metrics and inspect feature importances  
5. Apply RandomForestRegressor to a regression problem

# 1. Introduction to Random Forests

Random Forests combine multiple decision trees trained on different subsets of data to reduce variance and improve generalization.

# 2. Ensembling and Bagging

Bagging (Bootstrap AGGregatING): each tree is trained on a random sample of the data taken with replacement.

Feature subsampling: at each split, only a random subset of features is considered, which increases tree diversity.

Variance reduction: by averaging (in regression) or voting (in classification) across many independent trees, overall predictions become more stable.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
```

## Load example data
```python
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
```

## Split into train/test
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

## Train RandomForest
```python
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
```

## Predict and evaluate
```python
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
