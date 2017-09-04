# Exploring Kaggle Expedia Competition dataset based on https://www.dataquest.io/blog/kaggle-tutorial/

import pandas as pd
import random
import ml_metrics as metrics

data_dir = '/home/hungap/data/temp/kaggle/'

data_store_file = data_dir + '_hotel_search.hdf5'

# Finding correlations
# The closer it is to 1, the stronger the (linear) correlation
# For this training dataset, there is no strong linear correlation between hotel_cluster and other variables
train.corr()['hotel_cluster']

train = pd.read_csv(data_dir + "train.csv")
# Check dataset sizes
train.shape
train.columns.values
# Explore
train.head(5)

# Convert to date_time
train['date_time'] = pd.to_datetime(train['date_time'])
train["year"] = train["date_time"].dt.year
train["month"] = train["date_time"].dt.month

# Count by hotel cluster
train["hotel_cluster"].value_counts()
# Out[24]:
# 91    1043720
# 41     772743
# 48     754033
# 64     704734
# 65     670960
# 5      620194
# 98     589178
# 59     570291

# Down-sample using 10000 users and create train and test dataset
sel_user_ids = random.sample(set(train.user_id.unique()), 10000)
sel_train = train[train.user_id.isin(sel_user_ids)]
train_small = sel_train[((sel_train.year == 2013) | ((sel_train.year == 2014) & (sel_train.month < 8)))]
test_small = sel_train[((sel_train.year == 2014) & (sel_train.month >= 8))]
test_small = test_small[test_small.is_booking == True]

# Storing temporary dataset (for efficiency) in hdf5 format (https://dzone.com/articles/quick-hdf5-pandas)
store = pd.HDFStore(data_store_file)
store.put('train_small', train_small, format='table', data_columns=True)
store.put('test_small', test_small, format='table', data_columns=True)
store.close()

#########
###  Classifier 1 - Use the top 5 most common hotel clusters
##########
# Get the ids of the top 5 most common hotel clusters
most_common_clusters = list(train.hotel_cluster.value_counts().head().index)
# For each test sample, we provide 5 predictions i.e. 5 most common clusters
predictions = [most_common_clusters for i in range(len(test_small))]

# Convert target to a list to be used in the mapk method
target = [[l] for l in test_small["hotel_cluster"]]
metrics.mapk(target, predictions, k=5) # k = number of predictions??


