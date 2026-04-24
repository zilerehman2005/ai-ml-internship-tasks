# Task 1: Iris Dataset Exploration & Visualization

## Objective

The objective of this task is to perform Exploratory Data Analysis (EDA) on the Iris dataset in order to understand data patterns, feature relationships, and statistical properties.

## Dataset

* **Name:** Iris Dataset
* **Source:** Loaded using the Seaborn library
* **Description:**
  The dataset contains measurements of iris flowers with the following features:

  * Sepal Length
  * Sepal Width
  * Petal Length
  * Petal Width
  * Species (Setosa, Versicolor, Virginica)


## Methodology

### Data Loading and Inspection

* Loaded the dataset using pandas
* Examined dataset shape and column names
* Displayed initial records using `head()`

### Data Understanding

* Used `info()` to inspect data types and structure
* Used `describe()` to generate summary statistics
* Checked for missing values

### Data Visualization

To better understand the dataset, the following visualizations were created:

* **Scatter Plots**
  Used to analyze relationships between features and observe class separation

* **Histograms**
  Used to examine the distribution of individual features

* **Box Plots**
  Used to detect outliers and understand spread of data

* **Pair Plot**
  Used to visualize relationships between all feature combinations


## Tools and Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn


## Key Insights

* Petal length and petal width are the most significant features for distinguishing species
* Setosa class is clearly separable from the other two classes
* Minor outliers are present, particularly in sepal width
* Strong correlation exists between petal-related features


