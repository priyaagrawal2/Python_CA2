import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 #----------------EXPLORATORY DATA ANALYSIS-----------------
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

#check duplicate row present in each column
duplicate_rows = df.duplicated().sum()
print("\nDuplicate rows present in dataset: ",duplicate_rows)

#Histogram plot from Time column
plt.figure(figsize = (12,6))
sns.histplot(df['Time'],bins = 20,kde = True)
plt.title("Histogram plot")
plt.show()

#Box - plot
sns.boxplot(x = df['Time'])
plt.title("Box-plot on time")
# plt.show()
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

print("\nOutliers: ", outliers)



#Correaltion matrix and heatmap
correlation_matrix = df.corr(numeric_only = True)
print(correlation_matrix)
plt.figure(figsize=(6,6))
sns.heatmap(correlation_matrix, annot = True, cmap = "coolwarm",fmt = ".2f",linewidths = 0.8)
plt.title("Correlation Heatmap")
plt.show()

print("EDA completed")

#--------------------DATA CLEANING ---------------------


#There are no duplicate Rows in the data set

# Dropped the rows with too many missing values
df.dropna(thresh=0.8 * df.shape[1], inplace=True)
print("\nRows with too many missing values are dropped\n")

#Cleaning numeric columns
df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
df['Cranial_Capacity'] = pd.to_numeric(df['Cranial_Capacity'], errors='coerce')
df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
print("\ncolumns containg numeric values are cleaned\n")

#Making column names consistant
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('&', 'and')

#Removing duplicates
df.drop_duplicates(inplace=True)

#Check null values present in the column
null_values = df.isnull().sum()[df.isnull().sum() > 0]
print("\nNo. of null values present in the columns: ",null_values)

#To remove these null values for different columns 
df['Cranial_Capacity'] = df['Cranial_Capacity'].fillna(df['Cranial_Capacity'].mean())
df['Height'] = df['Height'].fillna(df['Height'].mean())
df['Diet'] = df['Diet'].fillna(df['Diet'].mode())
df['Sexual_Dimorphism'] = df['Sexual_Dimorphism'].fillna(df['Sexual_Dimorphism'].mode())
df['Migrated'] = df['Migrated'].fillna(df['Migrated'].mode())

#Now again check if there is any missing value
null_values = df.isnull().sum()[df.isnull().sum() > 0]
print("\nNo. of null values present in the columns: ",null_values)

print("data cleaning completed")

#----------------------DATA VISUALIZATION -----------------------

#configure the style and appearance of plot
sns.set_theme(style="whitegrid", palette="pastel")

#Histogram plot for column Cranial_Capacity
sns.histplot(df['Cranial_Capacity'], kde=True,  color='mediumslateblue')
plt.title("Histogram plot for Cranial Capacity")
plt.show()

#Some box plots
plt.figure(figsize=(12, 5))
# First plot: Cranial Capacity
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Cranial_Capacity'], color='skyblue')
plt.title("Cranial Capacity")
plt.ylabel("Cranial Capacity")

# Second plot: Height
plt.subplot(1, 2, 2)
sns.boxplot(y=df['Height'], color='lightgreen')
plt.title("Height")
plt.ylabel("Height")
plt.show()
#for showing 2 or more box plots in one we use subplot function

#Bar plot for average height for good visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Zone', y='Height', estimator='mean', palette='plasma',hue='Zone')
plt.title("Average Height by Zone")
plt.ylabel("Height in (cm)")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
