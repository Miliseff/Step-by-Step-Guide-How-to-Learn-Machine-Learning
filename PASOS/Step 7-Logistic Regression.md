# Example: Loan Approval

We'll use a small synthetic dataset to predict whether a loan application will be approved (1) or not (0) based on two features:

income (in thousands of euros)
loan_amount (in thousands of euros)

Step 1: Import Libraries
python
Copiar código
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
Step 2: Create Synthetic Dataset

```python
# Create a synthetic dataset
data = {
    'income': [30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    'loan_amount': [5, 15, 10, 25, 35, 45, 50, 60, 70, 80],
    'approved': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
```
## Convert to a pandas DataFrame
```python
df = pd.DataFrame(data)
```

Step 3: Visualize the Data

## Visualize the data
```python
plt.scatter(df['income'], df['loan_amount'], c=df['approved'], cmap='bwr')
plt.xlabel('Income (in thousands of euros)')
plt.ylabel('Loan Amount (in thousands of euros)')
plt.title('Loan Approval')
plt.show()
```
Step 4: Prepare Data for the Model

## Separate features and the target variable
```python
X = df[['income', 'loan_amount']]
y = df['approved']
```
## Split the dataset into training and testing sets
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
Step 5: Train the Logistic Regression Model

## Create the model
```python
model = LogisticRegression()
```
## Train the model
```python
model.fit(X_train, y_train)
```
Step 6: Make Predictions

## Make predictions on the test set
```python
y_pred = model.predict(X_test)
```

Step 7: Evaluate the Model

## Calculate accuracy
```python
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```
## Confusion Matrix
```python
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
```

Explanation of Results
Accuracy: Accuracy is the proportion of correct predictions out of the total predictions. An accuracy close to 1 indicates good model performance.
Confusion Matrix: The confusion matrix shows the number of true positives, true negatives, false positives, and false negatives. It is useful for evaluating model performance beyond simple accuracy.

## Complete Code
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Create a synthetic dataset
data = {
    'income': [30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
    'loan_amount': [5, 15, 10, 25, 35, 45, 50, 60, 70, 80],
    'approved': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

# Convert to a pandas DataFrame
df = pd.DataFrame(data)

# Visualize the data
plt.scatter(df['income'], df['loan_amount'], c=df['approved'], cmap='bwr')
plt.xlabel('Income (in thousands of euros)')
plt.ylabel('Loan Amount (in thousands of euros)')
plt.title('Loan Approval')
plt.show()

# Separate features and the target variable
X = df[['income', 'loan_amount']]
y = df['approved']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
```

Conclusion
This simple example shows how to use logistic regression to predict loan approval based on income and loan amount. We have learned how to prepare data, train a model, make predictions, and evaluate the model's performance using Python and libraries like scikit-learn.
