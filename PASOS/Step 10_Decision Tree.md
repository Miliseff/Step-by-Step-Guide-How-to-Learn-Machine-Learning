# Class: Decision Tree in Python

## 1. Introduction to Decision Trees

A **Decision Tree** is a supervised learning algorithm that can be used for both classification and regression tasks. Decision Trees are structured like a flowchart where each node represents a feature or attribute, each branch represents a decision or rule, and each leaf node represents the outcome or class label.

The algorithm splits the data at each node based on certain conditions, aiming to maximize the information gain or reduce the impurity (e.g., Gini index or entropy) at each step.

### How Does a Decision Tree Work?

1. **Selecting the Best Feature**: The algorithm selects a feature that best separates the data into distinct classes or values. This is usually done by calculating **information gain** or **Gini impurity**.
2. **Splitting the Data**: Based on the chosen feature, the data is split into subsets.
3. **Repeating the Process**: The algorithm recursively repeats the process on each subset, creating branches and nodes until a stopping criterion is met (e.g., maximum depth, minimum samples).
4. **Making a Prediction**: Once the tree is built, it can be used to classify new samples or make predictions by following the path of decisions in the tree.

---

## 2. Key Concepts

- **Gini Impurity**: Measures the likelihood of incorrectly classifying a randomly chosen element in the dataset. Lower Gini values indicate purer splits.
- **Entropy and Information Gain**: Entropy measures the disorder or impurity, and information gain quantifies the reduction in entropy after a dataset is split on a feature.
- **Max Depth**: Limits the depth of the tree, preventing it from becoming too complex and overfitting the data.
- **Minimum Samples Split**: Defines the minimum number of samples required to split a node.

---

## Advantages and Disadvantages of Decision Trees

- **Advantages**:
  - Easy to interpret and visualize.
  - Can handle both numerical and categorical data.
  - Requires little data preprocessing (e.g., no need to normalize or scale data).
  
- **Disadvantages**:
  - Prone to overfitting if not properly tuned.
  - Can become biased with imbalanced data.
  - Sensitive to small variations in the data (high variance).

---

## 3. Implementing Decision Trees in Python

We will use the `scikit-learn` library to implement Decision Tree models with examples for both classification and regression.

---

### Example 1: Decision Tree for Classification

In this example, we will use the **Iris** dataset to classify flower species based on features such as sepal length, sepal width, petal length, and petal width.

#### Step 1: Import Necessary Libraries

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree
```

#### Step 2: Load and Split the Dataset

```python
# Load the dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### Step 3: Create and Train the Decision Tree Classifier

```python
# Create a Decision Tree Classifier with a maximum depth of 3 to prevent overfitting
clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# Train the model
clf.fit(X_train, y_train)
```

#### Step 4: Make Predictions and Evaluate the Model

```python
# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

#### Step 5: Visualize the Decision Tree

```python
# Visualize the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```

---

### Example 2: Decision Tree for Regression

In this example, we will use a synthetic dataset to predict house prices based on house size.

#### Step 1: Generate Synthetic Data

```python
# Generate example data
np.random.seed(0)
X = np.random.rand(100, 1) * 100  # House sizes in square meters
y = X * 5 + (np.random.randn(100, 1) * 50)  # House prices with some noise
```

#### Step 2: Split the Data

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### Step 3: Create and Train the Decision Tree Regressor

```python
from sklearn.tree import DecisionTreeRegressor

# Create a Decision Tree Regressor with a maximum depth of 4 to avoid overfitting
regressor = DecisionTreeRegressor(max_depth=4, random_state=42)

# Train the model
regressor.fit(X_train, y_train)
```

#### Step 4: Make Predictions and Evaluate the Model

```python
# Make predictions
y_pred = regressor.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
```

#### Step 5: Visualize the Decision Tree Predictions

```python
# Plot the regression tree predictions
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Actual data")
plt.scatter(X_test, y_pred, color="red", label="Predicted data")
plt.xlabel("House Size (m²)")
plt.ylabel("House Price")
plt.legend()
plt.title("Decision Tree Regression - House Price Prediction")
plt.show()
```

---

## 4. Hyperparameter Tuning for Decision Trees

To improve the performance of Decision Trees, you can experiment with hyperparameters such as:
- **Max Depth**: Limits the depth of the tree.
- **Min Samples Split**: Minimum number of samples required to split a node.
- **Min Samples Leaf**: Minimum number of samples required at a leaf node.

You can use `GridSearchCV` to automate this process:

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Set up GridSearchCV
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)
```

---

## Conclusion

Decision Trees are versatile models that can be used for both classification and regression tasks. They are easy to interpret and can handle various types of data. However, they are prone to overfitting and may require pruning or tuning to achieve optimal performance.

This class provided an overview of Decision Trees, how they work, and practical examples for classification and regression. Would you like to explore other tree-based models, such as Random Forests or Gradient Boosting? 

Happy coding! 🎉
``` 
