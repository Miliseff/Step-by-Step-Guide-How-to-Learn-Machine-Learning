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
