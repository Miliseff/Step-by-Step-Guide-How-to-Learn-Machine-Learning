# Introduction to Linear Regression 



## Learning objectives

1. Analyze a Pandas Dataframe.
2. Create Seaborn plots for Exploratory Data Analysis.
3. Train a Linear Regression Model using Scikit-Learn.


## Introduction 
This lab is an introduction to linear regression using Python and Scikit-Learn.  This lab serves as a foundation for more complex algorithms and machine learning models that you will encounter in the course. You will train a linear regression model to predict housing price.

Each learning objective will correspond to a __#TODO__ in the [student lab notebook](../labs/intro_linear_regression.ipynb) -- try to complete that notebook first before reviewing this solution notebook.


### Import Libraries

Importing Pandas, a data processing and CSV file I/O libraries
```Python
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn is a Python data visualization library based on matplotlib. 
%matplotlib inline   
```
###  Load the Dataset

You will use the [USA housing prices](https://www.kaggle.com/kanths028/usa-housing) dataset found on Kaggle.  The data contains the following columns:

* 'Avg. Area Income': Avg. Income of residents of the city house is located in.
* 'Avg. Area House Age': Avg Age of Houses in same city
* 'Avg. Area Number of Rooms': Avg Number of Rooms for Houses in same city
* 'Avg. Area Number of Bedrooms': Avg Number of Bedrooms for Houses in same city
* 'Area Population': Population of city house is located in
* 'Price': Price that the house sold at
* 'Address': Address for the house


 Next, you read the dataset into a Pandas dataframe.
```Python
df_USAhousing = pd.read_csv('../USA_Housing_toy.csv')
```

 Show the first five row.
```Python
df_USAhousing.head()
```



Let's check for any null values.
The isnull() method is used to check and manage NULL values in a data frame.
```Python
df_USAhousing.isnull().sum()
```


Let's check for any null values.
Pandas describe() is used to view some basic statistical details of a data frame or a series of numeric values.
```Python
df_USAhousing.describe()
```

 Pandas info() function is used to get a concise summary of the dataframe.
```Python
df_USAhousing.info()
```

Let's take a peek at the first and last five rows of the data for all columns.
```Python
print(df_USAhousing,5) # TODO 1
```

## Exploratory Data Analysis (EDA)

Let's create some simple plots to check out the data!  

 Plot pairwise relationships in a dataset. By default, this function will create a grid of Axes such that each numeric variable in data will be
 shared across the y-axes across a single row and the x-axes across a single column.
```Python
sns.pairplot(df_USAhousing)
```


 It is used basically for univariant set of observations and visualizes it through a histogram i.e. only one observation
 and hence you choose one particular column of the dataset.
```Python
sns.displot(df_USAhousing['Price'])
```

The heatmap is a way of representing the data in a 2-dimensional form. The data values are represented as colors in the graph.
 The goal of the heatmap is to provide a colored visual summary of information.
```Python
sns.heatmap(df_USAhousing.corr(numeric_only=True)) # TODO 2
```

## Training a Linear Regression Model

Regression is a supervised machine learning process.  It is similar to classification, but rather than predicting a label, you try to predict a continuous value.   Linear regression defines the relationship between a target variable (y) and a set of predictive features (x).  Simply stated, If you need to predict a number, then use regression. 

Let's now begin to train your regression model! You will need to first split up your data into an X array that contains the features to train on, and a y array with the target variable, in this case the Price column. You will toss out the Address column because it only has text info that the linear regression model can't use.

### X and y arrays

Next, let's define the features and label.  Briefly, feature is input; label is output. This applies to both classification and regression problems.
```Python
X = df_USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = df_USAhousing['Price']
```

## Train - Test - Split

Now let's split the data into a training set and a testing set. You will train out model on the training set and then use the test set to evaluate the model.  Note that you are using 40% of the data for testing.  

What is Random State? 
If an integer for random state is not specified in the code, then every time the code is executed, a new random value is generated and the train and test datasets will have different values each time.  However, if a fixed value is assigned -- like random_state = 0 or 1 or 101 or any other integer, then no matter how many times you execute your code the result would be the same, e.g. the same values will be in the train and test datasets.  Thus, the random state that you provide is used as a seed to the random number generator. This ensures that the random numbers are generated in the same order. 


Import train_test_split function from sklearn.model_selection
```Python
from sklearn.model_selection import train_test_split
```

Split up the data into a training set
```Python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
```

## Creating and Training the Model

Import LinearRegression function from sklearn.model_selection
```Python
from sklearn.linear_model import LinearRegression
```

LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets
in the dataset, and the targets predicted by the linear approximation.
```Python
lm = LinearRegression()
```

# Train the Linear Regression Classifer
```Python
lm.fit(X_train,y_train) # TODO 3
```


## Model Evaluation

Let's evaluate the model by checking out it's coefficients and how you can interpret them.

print the intercept
```Python
print(lm.intercept_)
```

Pandas DataFrame is two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns).
```Python
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
```

Interpreting the coefficients:

- Holding all other features fixed, a 1 unit increase in **Avg. Area Income** is associated with an **increase of \$21.52 **.
- Holding all other features fixed, a 1 unit increase in **Avg. Area House Age** is associated with an **increase of \$164883.28 **.
- Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Rooms** is associated with an **increase of \$122368.67 **.
- Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Bedrooms** is associated with an **increase of \$2233.80 **.
- Holding all other features fixed, a 1 unit increase in **Area Population** is associated with an **increase of \$15.15 **.

## Predictions from your Model

Let's grab predictions off your test set and see how well it did!


Predict values based on linear model object.
```Python
predictions = lm.predict(X_test)
```

 Scatter plots are widely used to represent relation among variables and how change in one affects the other.
```Python
plt.scatter(y_test,predictions)
```

**Residual Histogram**

 It is used basically for univariant set of observations and visualizes it through a histogram i.e. only one observation
and hence you choose one particular column of the dataset.
```Python
sns.displot((y_test-predictions),bins=50);
```

## Regression Evaluation Metrics


Here are three common evaluation metrics for regression problems:

**Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:

$$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$

**Mean Squared Error** (MSE) is the mean of the squared errors:

$$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$

**Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:

$$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$

Comparing these metrics:

- **MAE** is the easiest to understand, because it's the average error.
- **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
- **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.

All of these are **loss functions**, because you want to minimize them.


Importing metrics from sklearn
```Python
from sklearn import metrics
```

 Show the values of MAE, MSE, RMSE
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
