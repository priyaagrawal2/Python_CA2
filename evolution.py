import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
#file ko read/load krna
df = pd.read_csv(r"C:\Users\mahen\Downloads\Evolution_DataSet_python.csv")

#Reading first 10 rows
print("First 10 rows of dataset: ",df.head(10))

#To get dataset information
print("Information about Evolution_dataset: ",df.info())

#Summary statistics of dataset
print("Discription: ",df.describe())

#Null values column wise
print("Null values present in dataset: ",df.isnull().sum())

#Null values in entire dataset
print("Total null values: ",df.isnull().sum().sum())

#Unique values present in dataset
unique_values = df.nunique()
print("Unique values present in dataset: ",unique_values)

#Unique values present in a column
unique_v = df['Jaw_Shape'].unique()
print("Unique values present in column name Jaw_Shape: ",unique_v)

#Data Types 
print("Data types present in dataset: ",df.dtypes)

#Duplicate rows present in each column
duplicate_rows = df.duplicated().sum()
print("Duplicate rows present in dataset: ",duplicate_rows)

#Box - plot
sns.boxplot(x = df['Time'])
plt.title("Box-plot on time")
plt.show()
#By seeing the box plot of time column we get to know this is right - skewed data(use IQR score)

#Outliers that are present in dataset in column 'Time'
Q1 = df['Time'].quantile(0.25)
Q3 = df['Time'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
print("Lower bound for the column time: ",lower_bound)
print("Upper bound for the column time: ",upper_bound)

outliers = ((df['Time'] < lower_bound) | (df['Time'] > upper_bound))

print("Outliers: ", outliers)



#Correaltion matrix and heatmap
correlation_matrix = df.corr(numeric_only = True)
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot = True, cmap = "Reds",fmt = ".2f",linewidths = 0.8)
plt.title("Correlation Heatmap")
plt.show()
