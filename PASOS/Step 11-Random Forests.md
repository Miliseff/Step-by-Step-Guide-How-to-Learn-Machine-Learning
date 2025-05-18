# Random Forests 

**Learning Objectives**

1. Understand the idea of ensembling and bagging  
2. Build a `RandomForestClassifier` with scikit-learn  
3. Use Out-of-Bag (OOB) samples to estimate test error  
4. Tune the key hyperparameters: `n_estimators`, `max_depth`, `max_features`  
5. Check feature importances and standard metrics  
6. Apply `RandomForestRegressor` to a regression task  
7. Know the main strengths, limits, and best practices  

---

# 1. Introduction to Random Forests

A Random Forest is a collection of decision trees.  
Every tree is trained on a slightly different slice of the data, then all trees vote (classification) or average (regression).  
Because the trees disagree a little, their errors cancel out and the final model is more stable.

---

# 2. Ensembling and Bagging

Bagging (Bootstrap Aggregating): each tree receives a random sample of the training rows **with replacement**.  

Feature subsampling: at every split, the tree looks at only a random subset of columns.  

Variance reduction: many diverse trees → one smooth and reliable prediction.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
````

# Load example data

```python
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
```

# Split into train / test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

# Train RandomForest (with OOB score)

```python
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    max_features="sqrt",
    oob_score=True,
    random_state=42
)
rf.fit(X_train, y_train)
print("OOB accuracy:", rf.oob_score_)
```

# Predict and evaluate on the test set

```python
y_pred = rf.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

# 3. Hyperparameter Tuning (Grid Search)

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 6, 12],
    "max_features": ["sqrt", "log2"]
}

gs = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
print("Best CV score:", gs.best_score_)
```

---

# 4. Feature Importance

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(
    rf.feature_importances_,
    index=load_iris().feature_names
).sort_values()

plt.barh(importances.index, importances.values)
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.show()
```

---

# 5. RandomForestRegressor (Regression Example)

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

X_reg, y_reg = fetch_california_housing(return_X_y=True)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=0
)

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    random_state=0
)
rf_reg.fit(Xr_train, yr_train)

yr_pred = rf_reg.predict(Xr_test)
print("MSE:", mean_squared_error(yr_test, yr_pred))
```

---

# 6. Advantages, Limitations, Best Practices

**Pros**

* Works with many rows and columns, little preprocessing
* Handles missing values and outliers quite well
* Gives built-in feature importance

**Cons**

* Less interpretable than a single tree
* Larger models → more memory and slower predictions
* Can still overfit if `n_estimators` is too low and `max_depth` too high

**Tips**

* Start with `n_estimators = 100-200`
* Use `max_features="sqrt"` (classification) or `"auto"` (regression)
* Limit `max_depth` if you see overfitting
* Turn on `oob_score=True` to get a quick unbiased error estimate



