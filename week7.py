import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import os # Used for the 'time-series' simulation
import warnings

# Set a consistent style for attractive statistical plots
sns.set_theme(style="whitegrid")

print("Starting Data Analysis Assignment...")
print("-" * 50)


# ==============================================================================
# Task 1: Load and Explore the Dataset
# ==============================================================================

# Use exception handling for robust loading (as required by instructions)
try:
    # 1. Load the dataset (Iris)
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # Add the target column for categorization (Species)
    df['species'] = iris.target_names[iris.target]

    print("Task 1.1: Dataset successfully loaded.")

    # 2. Simulate Missing Values for Cleaning Task
    # (If using a real CSV, this step can be skipped)
    np.random.seed(42)
    df.loc[np.random.choice(df.index, size=5, replace=False), 'sepal length (cm)'] = np.nan
    df.loc[np.random.choice(df.index, size=3, replace=False), 'petal width (cm)'] = np.nan

except Exception as e:
    print(f"Error during dataset loading: {e}")
    # Exit gracefully if loading fails
    exit()

# Display the first few rows
print("\nTask 1.2: First 5 rows of the DataFrame (df.head()):")
print(df.head())
print("-" * 50)

# Explore the structure (data types and non-null counts)
print("Task 1.3: DataFrame Structure and Missing Values (df.info() & df.isnull().sum()):")
df.info()

# Count missing values
print("\nInitial Missing Value Count:")
print(df.isnull().sum())
print("-" * 50)

# 3. Clean the dataset by handling missing values
# Fill 'sepal length (cm)' NaNs with the mean
df['sepal length (cm)'].fillna(df['sepal length (cm)'].mean(), inplace=True)
# Fill 'petal width (cm)' NaNs with the median (more robust to potential outliers)
df['petal width (cm)'].fillna(df['petal width (cm)'].median(), inplace=True)

print("Task 1.4: Missing values cleaned (imputed with mean/median).")
print("Missing Value Count After Cleaning:")
print(df.isnull().sum())
print("-" * 50)


# ==============================================================================
# Task 2: Basic Data Analysis
# ==============================================================================

# 1. Compute basic statistics using .describe()
print("Task 2.1: Descriptive Statistics of Numerical Columns (df.describe()):")
print(df.describe())
print("-" * 50)

# 2. Perform groupings and aggregation
# Group by the categorical column 'species' and compute the mean of all numerical columns
species_mean = df.groupby('species').mean()
print("Task 2.2: Mean of all features grouped by 'species':")
print(species_mean)
print("-" * 50)

# 3. Identify patterns/findings
print("Task 2.3: Key Findings:")
print(" - Setosa has the smallest petals (both length and width).")
print(" - Virginica has the largest petals and sepals overall.")
print(" - Versicolor features are intermediate between the other two species.")
print("-" * 50)


# ==============================================================================
# Task 3: Data Visualization
# ==============================================================================

# Set up the figure for a multi-plot layout
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Data Visualization of Iris Dataset', fontsize=16)

# --- Plot 1: Line Chart (Simulated Trend) ---
# Since Iris is not time-series, we simulate a trend using cumulative mean.
df['cumulative_mean_sepal_length'] = df['sepal length (cm)'].expanding().mean()
axes[0, 0].plot(df.index, df['cumulative_mean_sepal_length'], color='darkblue', linewidth=2)
axes[0, 0].set_title('1. Simulated Trend: Cumulative Mean of Sepal Length', fontsize=12)
axes[0, 0].set_xlabel('Observation Index')
axes[0, 0].set_ylabel('Cumulative Mean Sepal Length (cm)')
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# --- Plot 2: Bar Chart (Comparison Across Categories) ---
# Compare the average petal length across the three species
sns.barplot(
    x=species_mean.index,
    y=species_mean['petal length (cm)'],
    ax=axes[0, 1],
    palette='Pastel1'
)
axes[0, 1].set_title('2. Average Petal Length by Species', fontsize=12)
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Average Petal Length (cm)')

# --- Plot 3: Histogram (Distribution) ---
# Visualize the distribution of Sepal Width
sns.histplot(
    df['sepal width (cm)'],
    kde=True, # Add a kernel density estimate line
    bins=15,
    color='orange',
    ax=axes[1, 0]
)
axes[1, 0].set_title('3. Distribution of Sepal Width (cm)', fontsize=12)
axes[1, 0].set_xlabel('Sepal Width (cm)')
axes[1, 0].set_ylabel('Frequency')

# --- Plot 4: Scatter Plot (Relationship between variables) ---
# Visualize the correlation between sepal length and petal length, colored by species
sns.scatterplot(
    x='sepal length (cm)',
    y='petal length (cm)',
    hue='species',
    data=df,
    s=70, # size of points
    ax=axes[1, 1]
)
axes[1, 1].set_title('4. Scatter Plot: Sepal Length vs. Petal Length (by Species)', fontsize=12)
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')
axes[1, 1].legend(title='Species', loc='upper left')

# Adjust layout to prevent overlapping titles/labels
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust main title space

# Display the final plots
plt.show()

print("-" * 50)
print("Assignment tasks completed successfully. Dataframe analysis printed, and plots displayed.")