# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 09:42:13 2023

@author: alothimen
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer

# =============================================================================
#                       Exploratory Data Analysis
# =============================================================================

#Load the dataset
df= pd.read_csv('kidney_disease.csv')

# Diagnose data for cleaning
print(".head() returns the first few rows: \n",df.head())


print("\n.tail() returns the last few rows: \n",df.tail())

print("\n.info() shows information on each of the columns: \n", df.info())

print("\n.shape() returns the number of rows and columns of the DataFrame: \n",df.shape)

print("Column names: \n",df.columns)

# Check for missing Values
print("Count of missing values in each column: \n",df.isnull().sum())

# Explore descriptive statistics
print("Summary statistics for numerical columns: \n", df.describe()) 

# =============================================================================
#                           DATA CLEANING
# =============================================================================

# Rename column names to make it more user-friendly
df.columns = ['id', 'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'anemia', 'class']

# Replace some incorrect values
df['diabetes_mellitus'].replace(to_replace={'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True)
df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace='\tno', value='no')
df['class'] = df['class'].replace(to_replace={'ckd\t': 'ckd', 'notckd': 'notckd'})

# Convert 'Specific gravity', 'Albumin', 'Sugar' to object (nominal) type
df['specific_gravity'] = df['specific_gravity'].astype('object')
df['albumin'] = df['albumin'].astype('object')
df['sugar'] = df['sugar'].astype('object')

#Update
# Convert 'ckd' and 'notckd' to 0 and 1 in the 'class' column
df['class'] = df['class'].map({'notckd': 0, 'ckd': 1})

# Convert the 'class' column to int64
df['class'] = df['class'].astype('int64')

#df['class'] = df['class'].astype('object')
# After running line 66, the visualization for 'class' count plot 
# will not appear since it is no longer an object data type

# Extracting categorical and numerical columns
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']

# Convert 'packed_cell_volume', 'white_blood_cell_count', and 'red_blood_cell_count' to numeric
df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')

# Replace missing values in numerical columns with the most frequent value
df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.mode().iloc[0]))

# Replace missing values in categorical columns with the most frequent value for each column
df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode().iloc[0]))

# Save the cleaned and imputed DataFrame to a new CSV file
df.to_csv('cleaned_kidney_disease.csv', index=False)

# =============================================================================
#                         VISUALIZATION
# =============================================================================

# Checking numerical features distribution (excluding 'id' column)
plt.figure(figsize=(20, 15))

# Define the list of numeric columns
numeric_cols = ['age', 'blood_pressure', 'blood_glucose_random', 'blood_urea', 
                'serum_creatinine', 'sodium', 'potassium', 'hemoglobin', 
                'packed_cell_volume', 'white_blood_cell_count','red_blood_cell_count']

for plotnumber, column in enumerate(numeric_cols, start=1):
    ax = plt.subplot(3, 5, plotnumber)
    sns.histplot(df[column], color='red', kde=True, stat="density", linewidth=0)
    plt.xlabel(column)

plt.tight_layout()
plt.show()

# Heatmap of the correlation matrix (excluding non-numeric columns and 'id')
plt.figure(figsize=(25, 20))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.show()

# Categorical columns plot
cat_cols_only = [col for col in cat_cols if col not in numeric_cols]

# Set the number of rows and columns for subplots
num_rows = 2
num_cols_per_row = len(cat_cols_only) // num_rows + (len(cat_cols_only) % num_rows > 0)

# Set the figure size
plt.figure(figsize=(30, 20))

# Iterate through categorical columns and create countplots
for i, column in enumerate(cat_cols_only, start=1):
    plt.subplot(num_rows, num_cols_per_row, i)
    sns.countplot(x=column, data=df, palette='viridis')
    plt.title(f'Countplot of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')

# Adjust layout for better visualization
plt.tight_layout()

# Show the plots
plt.show()

# Show information of each column after data cleaning
df.info()
# =============================================================================


