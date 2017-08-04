# Example of using pandas

!conda install scikit-learn
!conda install pandas

import os
import numpy as np
import pandas as pd

current_dir = os.path.realpath(".") + "/"

# Read cycles CSV
iris = pd.read_csv(current_dir + 'iris.csv')

# Show dataframe columns
iris.info()
iris.head(10)

iris['species'].unique() # List classes
len(iris) # Number of observations
iris.describe() # Statistical summary


iris.sort_values('species') # Sort by class name
iris.iloc[1, 1] = np.nan # Update a field to NaN
iris.head(10)
iris = iris.dropna() # Drop observation with NaN in one of the columns
len(iris) # Removed one observation

# Plot graphs
iris.plot.hist()
iris.plot.scatter(x='sepal_length', y='sepal_width')

# Observations subset
iris[iris['sepal_length'].between(5, 6)]
iris[iris['sepal_length'] < 6]

# Get Range
iris['sepal_length'].min()
iris['sepal_length'].max()

# Group By Aggregation
iris.groupby('species').count()
iris.groupby('species').mean()

# Merge datasets
symbols = pd.DataFrame(
        {"species" : ['setosa', 'versicolor', 'virginica'],
         "symbols": ['ST', 'VC', 'V']})
merged = pd.merge(iris, symbols, how='inner', on='species')
merged.head(10)

merged.to_csv(current_dir + 'iris1.csv')