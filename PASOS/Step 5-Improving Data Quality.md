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


# Use the chown command to change the ownership of the repository to user
!sudo chown -R jupyter:jupyter /home/jupyter/training-data-analyst

### Import Libraries
