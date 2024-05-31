# Improving Data Quality

**Learning Objectives**


1. Resolve missing values
2. Convert the Date feature column to a datetime format
3. Rename a feature column, remove a value from a feature column
4. Create one-hot encoding features
5. Understand temporal feature conversions 


## Introduction 

Recall that machine learning models can only consume numeric data, and that numeric data should be "1"s or "0"s.  Data is said to be "messy" or "untidy" if it is missing attribute values, contains noise or outliers, has duplicates, wrong data, upper/lower case column names, and is essentially not ready for ingestion by a machine learning algorithm.  

This notebook presents and solves some of the most common issues of "untidy" data.  Note that different problems will require different methods, and they are beyond the scope of this notebook.

```Python
# Use the chown command to change the ownership of the repository to user
!sudo chown -R jupyter:jupyter /home/jupyter/training-data-analyst
```

### Import Libraries
```Python
# Importing necessary tensorflow library and printing the TF version.
import tensorflow as tf

print("TensorFlow version: ",tf.version.VERSION)
```
```Python
import os
# Here we'll import Pandas and Numpy data processing libraries
import pandas as pd  
import numpy as np
from datetime import datetime
# Use matplotlib for visualizing the model
import matplotlib.pyplot as plt
# Use seaborn for data visualization
import seaborn as sns
%matplotlib inline
```

### Load the Dataset

The dataset is based on California's [Vehicle Fuel Type Count by Zip Code](https://data.ca.gov/dataset/vehicle-fuel-type-count-by-zip-codeSynthetic) report.  The dataset has been modified to make the data "untidy" and is thus a synthetic representation that can be used for learning purposes.  
```Python
# Creating directory to store dataset
if not os.path.isdir("../data/transport"):
    os.makedirs("../data/transport")
```
```Python
# ls shows the working directory's contents.
# Using the -l parameter will lists files with assigned permissions
!ls -l ../data/transport
```

### Read Dataset into a Pandas DataFrame
Next, let's read in the dataset just copied from the cloud storage bucket and create a Pandas DataFrame.  We also add a Pandas .head() function to show you the top 5 rows of data in the DataFrame. Head() and Tail() are "best-practice" functions used to investigate datasets.  

```Python

# Reading "untidy_vehicle_data_toy.csv" file using the read_csv() function included in the pandas library.
df_transport = pd.read_csv('../data/transport/untidy_vehicle_data_toy.csv')

# Output the first five rows.
df_transport.head()
```

### DataFrame Column Data Types

DataFrames may have heterogenous or "mixed" data types, that is, some columns are numbers, some are strings, and some are dates etc. Because CSV files do not contain information on what data types are contained in each column, Pandas infers the data types when loading the data, e.g. if a column contains only numbers, Pandas will set that column’s data type to numeric: integer or float.

Run the next cell to see information on the DataFrame.
```Python

# The .info() function will display the concise summary of an dataframe.
df_transport.info()
```

From what the .info() function shows us, we have six string objects and one float object. We can definitely see more of the "string" object values now!
```Python
# Let's print out the first and last five rows of each column.
print(df_transport,5)
```

### Summary Statistics 

At this point, we have only one column which contains a numerical value (e.g. Vehicles).  For features which contain numerical values, we are often interested in various statistical measures relating to those values. Note, that because we only have one numeric feature, we see only one summary stastic - for now.  
```Python

# We can use .describe() to see some summary statistics for the numeric fields in our dataframe.
df_transport.describe()
```

Let's investigate a bit more of our data by using the .groupby() function.
```Python

# The .groupby() function is used for spliting the data into groups based on some criteria.
grouped_data = df_transport.groupby(['Zip Code','Model Year','Fuel','Make','Light_Duty','Vehicles'])

 # Get the first entry for each month.
df_transport.groupby('Fuel').first()
```

### Checking for Missing Values

Missing values adversely impact data quality, as they can lead the machine learning model to make inaccurate inferences about the data. Missing values can be the result of numerous factors, e.g. "bits" lost during streaming transmission, data entry, or perhaps a user forgot to fill in a field.  Note that Pandas recognizes both empty cells and “NaN” types as missing values. 

#### Let's show the null values for all features in the DataFrame.
```Python
df_transport.isnull().sum()
```

To see a sampling of which values are missing, enter the feature column name.  You'll notice that "False" and "True" correpond to the presence or abscence of a value by index number.
```Python

print (df_transport['Date'])
print (df_transport['Date'].isnull())
```
```Python

print (df_transport['Make'])
print (df_transport['Make'].isnull())
```
```Python

print (df_transport['Model Year'])
print (df_transport['Model Year'].isnull())
```

### What can we deduce about the data at this point?

# Let's summarize our data by row, column, features, unique, and missing values.

```Python

# In Python shape() is used in pandas to give the number of rows/columns.
# The number of rows is given by .shape[0]. The number of columns is given by .shape[1].
# Thus, shape() consists of an array having two arguments -- rows and columns
print ("Rows     : " ,df_transport.shape[0])
print ("Columns  : " ,df_transport.shape[1])
print ("\nFeatures : \n" ,df_transport.columns.tolist())
print ("\nUnique values :  \n",df_transport.nunique())
print ("\nMissing values :  ", df_transport.isnull().sum().values.sum())
```

Let's see the data again -- this time the last five rows in the dataset.
```Python

# Output the last five rows in the dataset.
df_transport.tail()
```

### What Are Our Data Quality Issues?

1. **Data Quality Issue #1**:  
> **Missing Values**:
Each feature column has multiple missing values.  In fact, we have a total of 18 missing values.
2. **Data Quality Issue #2**: 
> **Date DataType**:  Date is shown as an "object" datatype and should be a datetime.  In addition, Date is in one column.  Our business requirement is to see the Date parsed out to year, month, and day.  
3. **Data Quality Issue #3**: 
> **Model Year**: We are only interested in years greater than 2006, not "<2006".
4. **Data Quality Issue #4**:  
> **Categorical Columns**:  The feature column "Light_Duty" is categorical and has a "Yes/No" choice.  We cannot feed values like this into a machine learning model.  In addition, we need to "one-hot encode the remaining "string"/"object" columns.
5. **Data Quality Issue #5**:  
> **Temporal Features**:  How do we handle year, month, and day?


#### Data Quality Issue #1:  
##### Resolving Missing Values

Most algorithms do not accept missing values.  Yet, when we see missing values in our dataset, there is always a tendency to just "drop all the rows" with missing values.  Although Pandas will fill in the blank space with “NaN", we should "handle" them in some way.

While all the methods to handle missing values is beyond the scope of this lab, there are a few methods you should consider.  For numeric columns, use the "mean" values to fill in the missing numeric values.  For categorical columns, use the "mode" (or most frequent values) to fill in missing categorical values. 

In this lab, we use the .apply and Lambda functions to fill every column with its own most frequent value.  You'll learn more about Lambda functions later in the lab.
Let's check again for missing values by showing how many rows contain NaN values for each feature column.

```Python
# The isnull() method is used to check and manage NULL values in a data frame.
# TODO 1a
df_transport.isnull().sum()
```

Run the cell to apply the lambda function.
```Python

# Here we are using the apply function with lambda.

# We can use the apply() function to apply the lambda function to both rows and columns of a dataframe.
# TODO 1b
df_transport = df_transport.apply(lambda x:x.fillna(x.value_counts().index[0]))
```

Let's check again for missing values.
```Python

# The isnull() method is used to check and manage NULL values in a data frame.
# TODO 1c
df_transport.isnull().sum()
```

#### Data Quality Issue #2:  
##### Convert the Date Feature Column to a Datetime Format
```Python

# The date column is indeed shown as a string object. We can convert it to the datetime datatype with the to_datetime() function in Pandas.
# TODO 2a
df_transport['Date'] =  pd.to_datetime(df_transport['Date'],
                              format='%m/%d/%Y')
```
```Python

# Date is now converted and will display the concise summary of an dataframe.
# TODO 2b
df_transport.info()
```
```Python

# Now we will parse Date into three columns that is year, month, and day.
df_transport['year'] = df_transport['Date'].dt.year
df_transport['month'] = df_transport['Date'].dt.month
df_transport['day'] = df_transport['Date'].dt.day

#df['hour'] = df['date'].dt.hour - you could use this if your date format included hour.
#df['minute'] = df['date'].dt.minute - you could use this if your date format included minute.

# The .info() function will display the concise summary of an dataframe.
df_transport.info()
```

Let's confirm the Date parsing. This will also give us a another visualization of the data.
```Python

# Here, we are creating a new dataframe called "grouped_data" and grouping by on the column "Make"
grouped_data = df_transport.groupby(['Make'])

# Get the first entry for each month.
df_transport.groupby('month').first()
```

Now that we have Dates as a integers, let's do some additional plotting.
```Python

# Here we will visualize our data using the figure() function in the pyplot module of matplotlib's library -- which is used to create a new figure.
plt.figure(figsize=(10,6))

# Seaborn's .jointplot() displays a relationship between 2 variables (bivariate) as well as 1D profiles (univariate) in the margins. This plot is a convenience class that wraps JointGrid.
sns.jointplot(x='month',y='Vehicles',data=df_transport)

# The title() method in matplotlib module is used to specify title of the visualization depicted and displays the title using various attributes.
plt.title('Vehicles by Month')
```

#### Data Quality Issue #3:  
##### Rename a Feature Column and Remove a Value.  

Our feature columns have different "capitalizations" in their names, e.g. both upper and lower "case".  In addition, there are "spaces" in some of the column names.  In addition, we are only interested in years greater than 2006, not "<2006".  
We can also resolve the "case" problem too by making all the feature column names lower case.

```Python

# Let's remove all the spaces for feature columns by renaming them.
# TODO 3a
df_transport.rename(columns = { 'Date': 'date', 'Zip Code':'zipcode', 'Model Year': 'modelyear', 'Fuel': 'fuel', 'Make': 'make', 'Light_Duty': 'lightduty', 'Vehicles': 'vehicles'}, inplace = True) 

# Output the first two rows.
df_transport.head(2)
```
