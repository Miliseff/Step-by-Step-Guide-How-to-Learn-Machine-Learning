# Class: Decision Tree in Python 

## 1. Introduction to Decision Trees

A **Decision Tree** is a supervised learning algorithm that can be used for both classification and regression tasks.  
Think of it like a simple *“20 Questions”* game: at each step you ask a yes/no (or numeric) question about the data, follow the branch, and keep asking until you reach an answer.  
Each **node** stores a question about one feature, each **branch** is the outcome of that question, and each **leaf** gives the final prediction.

### How Does a Decision Tree Work?

1. **Selecting the Best Feature** – choose the feature that best separates the data (highest information gain / lowest Gini impurity).  
2. **Splitting the Data** – divide the dataset into subsets according to the chosen feature’s rule.  
3. **Repeating the Process** – recursively repeat the procedure on each subset until a stopping rule is met (max depth, min samples, etc.).  
4. **Making a Prediction** – to predict for a new sample, start at the root and follow the questions down to a leaf; the label or value stored in that leaf is the output.

---

## 2. Key Concepts

- **Gini Impurity** – probability of mis-classifying a random sample in the node (lower is better).  
- **Entropy & Information Gain** – entropy measures disorder; information gain is the drop in entropy after a split.  
- **Max Depth** – largest number of splits from root to leaf; limits complexity.  
- **Minimum Samples Split / Leaf** – minimum samples required to create a branch or stay in a leaf.

---

## Advantages and Disadvantages of Decision Trees

**Advantages**  
- Easy to interpret and visualize.  
- Handles both numeric and categorical features.  
- Needs little preprocessing (no scaling or normalization).

**Disadvantages**  
- Can overfit if the tree grows too deep.  
- Sensitive to small changes in data (high variance).  
- Biased toward features with many possible splits.

---

## 3. Implementing Decision Trees in Python

Below are concise examples using `scikit-learn` for classification and regression, now with detailed inline comments.

### Example 1 – Classification (Iris dataset)

```python
# ---------------------------------------------------------------
# Decision-Tree Classification on the Iris flower dataset
# ---------------------------------------------------------------

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree   # for plotting

# 1) Load the dataset (features X and labels y)
iris = load_iris()
X, y = iris.data, iris.target           # X: 4 numeric features, y: 3 classes

# 2) Split into train and test sets (70 % training, 30 % testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# 3) Create the Decision Tree (limit depth to avoid overfitting)
clf = DecisionTreeClassifier(
    max_depth=3,        # maximum levels from root to leaf
    random_state=42
)

# 4) Train (fit) the model on the training data
clf.fit(X_train, y_train)

# 5) Predict the test set
y_pred = clf.predict(X_test)

# 6) Evaluate accuracy and show per-class metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("Classification report:\n", classification_report(y_test, y_pred))

# 7) Visualize the tree structure
plt.figure(figsize=(12, 8))
tree.plot_tree(
    clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True          # color nodes by class purity
)
plt.title("Decision Tree for Iris Classification")
plt.show()
````


### Example 2 – Regression (synthetic house-price data)

```python
# ---------------------------------------------------------------
# Decision-Tree Regression: predict house price from size
# ---------------------------------------------------------------

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1) Create synthetic data
np.random.seed(0)
X = np.random.rand(100, 1) * 100          # X: house size (m²) from 0-100
y = X * 5 + np.random.randn(100, 1) * 50  # y: price with noise

# 2) Train / test split (70 % / 30 %)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# 3) Build a Decision-Tree Regressor (depth limited to reduce variance)
reg = DecisionTreeRegressor(
    max_depth=4,    # deeper = more flexible but higher risk of overfitting
    random_state=42
)

# 4) Fit on training data
reg.fit(X_train, y_train)

# 5) Predict test set prices
y_pred = reg.predict(X_test)

# 6) Evaluate with Mean Squared Error (lower = better)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# 7) Plot actual vs. predicted
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Actual data")
plt.scatter(X_test, y_pred, color="red", label="Predicted (test)")
plt.xlabel("House Size (m²)")
plt.ylabel("Price")
plt.legend()
plt.title("Decision Tree Regression – House Price Prediction")
plt.show()
```

---

## 4. Hyperparameter Tuning

```python
# ---------------------------------------------------------------
# Grid-search for the best Decision-Tree parameters
# ---------------------------------------------------------------

from sklearn.model_selection import GridSearchCV

# Parameter grid to explore
param_grid = {
    "max_depth": [2, 4, 6, 8, 10],        # try shallow to deep trees
    "min_samples_split": [2, 5, 10],      # node must have ≥ this many samples to split
    "min_samples_leaf": [1, 2, 4]         # leaf must keep ≥ this many samples
}

# GridSearchCV will train one model per combination (cross-validated)
grid = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,               # 5-fold cross-validation
    scoring="accuracy"
)
grid.fit(X_train, y_train)

print("Best parameters found:", grid.best_params_)
print("Best cross-validated accuracy:", grid.best_score_)
```


## Conclusion

Decision Trees are intuitive, powerful baseline models for both classification and regression.
Keep them shallow or prune them to prevent overfitting, and use hyperparameter tuning to find the sweet spot between bias and variance.
For stronger performance on larger or noisier datasets, move to tree ensembles such as **Random Forests** or **Gradient Boosting**.


