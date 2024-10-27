# Lesson: K-Nearest Neighbors (KNN)

## 1. Introduction to K-Nearest Neighbors (KNN)

**K-Nearest Neighbors** is a supervised learning algorithm used for classification and regression. The basic idea behind KNN is simple: given a new data point, the algorithm finds the `K` closest points to it in the training data and makes a decision based on the labels or values of those points.

### How Does KNN Work?

1. **Choose a value for `K`**: `K` represents the number of nearest neighbors to consider when making a decision.
2. **Calculate the distance**: For each new data point, calculate the distance between this point and all points in the dataset. Common distance metrics include:
   - **Euclidean Distance**: \( \sqrt{\sum (x_i - y_i)^2} \)
   - **Manhattan Distance**: \( \sum |x_i - y_i| \)
3. **Identify the `K` closest neighbors**: Select the `K` training points closest to the new data point.
4. **Classification or Regression**:
   - **Classification**: The class that appears most frequently among the `K` neighbors is assigned to the new data point.
   - **Regression**: The average of the values of the `K` neighbors is calculated to obtain the prediction.

### Pros and Cons of KNN

- **Advantages**:
  - Simple to understand and implement.
  - Does not make assumptions about data distribution.
  - Effective for classification problems in datasets with few features.

- **Disadvantages**:
  - Computational complexity increases with the size of the dataset, as it needs to calculate distances for each new point.
  - Sensitive to the scale of the data (it’s common to normalize or standardize the data).
  - Not ideal for high-dimensional data, as performance decreases due to the “curse of dimensionality.”

---

## 2. Implementing KNN in Python

To illustrate KNN, let’s implement two practical examples using the `scikit-learn` library in Python:

1. **Classification Example**: Classify flower species from the Iris dataset.
2. **Regression Example**: Predict house prices based on size.

---

### Example 1: Classification with KNN (Iris Dataset)

The **Iris** dataset is widely used for classification. It contains features of three flower species (setosa, versicolor, virginica), like sepal and petal length and width.

Step 1: Import Necessary Libraries

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```

Step 2: Load the Dataset and Split it into Training and Testing Sets

Load the dataset
```python
iris = load_iris()
X = iris.data  # Features: lengths and widths
y = iris.target  # Labels: flower species
```
Split the data into training and testing sets
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Step 3: Create and Train the KNN Model
# Define the KNN model with K=3
```python
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)
```
# Train the model
```python
knn_classifier.fit(X_train, y_train)
```

Step 4: Make Predictions and Evaluate the Model

# Make predictions
```python
y_pred = knn_classifier.predict(X_test)
```
# Calculate accuracy
```python
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN model accuracy with K={k}: {accuracy * 100:.2f}%")
```



### Example 2: Regression with KNN (House Price Prediction)

Let’s use synthetic data to predict house prices based on house size.

#### Step 1: Import Necessary Libraries

```python
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as pltibraries
```


Step 2: Create Example Data

```python
# Example data
X = np.array([[150], [200], [250], [300], [350], [400]]).reshape(-1, 1)  # House sizes (in m²)
y = np.array([200000, 250000, 300000, 350000, 400000, 450000])  # House prices
```

```python
# Visualize the data
plt.scatter(X, y, color="blue")
plt.xlabel("House Size (m²)")
plt.ylabel("House Price")
plt.title("House Prices")
plt.show()
```
Step 3: Create and Train the KNN Model

# Create the KNN model with K=2 for regression
k = 2
knn_regressor = KNeighborsRegressor(n_neighbors=k)

# Train the model
knn_regressor.fit(X, y)
Step 4: Make Predictions
python
Copiar código
# Make a prediction for a 325 m² house
prediction = knn_regressor.predict([[325]])
print(f"Predicted price for a 325 m² house: ${prediction[0]:,.2f}")
3. Choosing the Value of K
The value of K directly affects model performance:

Small values of K (e.g., K=1 or K=3) tend to create models that fit closely to the training data, which can lead to overfitting.
Large values of K tend to make the model more generalized, which can lead to underfitting if important patterns are ignored.
A good way to choose the value of K is to try different values and evaluate model performance on a validation set.

python
Copiar código
# Example of testing different values of K
from sklearn.model_selection import cross_val_score

k_values = range(1, 11)
accuracies = []

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_classifier, X_train, y_train, cv=5)
    accuracies.append(scores.mean())

# Plot accuracy as a function of K
plt.plot(k_values, accuracies)
plt.xlabel("Value of K")
plt.ylabel("Average Accuracy")
plt.title("Choosing the Best K Value")
plt.show()
Conclusion
The K-Nearest Neighbors algorithm is easy to understand and implement and is useful for both classification and regression tasks in small-scale problems or well-distributed data. However, it has limitations in high-dimensional or large datasets due to its computational cost in the prediction phase.

This lesson provides a fundamental understanding of KNN and practical examples to apply it. Would you like to dive deeper into any specific aspect or do more practical exercises?

csharp
Copiar código

This Markdown file is organized, with clear headings, code blocks, and explanations. You can di