# Exploring Kaggle Expedia Competition dataset based on https://www.dataquest.io/blog/kaggle-tutorial/

import pandas as pd
import random
import ml_metrics as metrics

data_dir = os.path.realpath(".") + "/"

data_dir = '/home/hungap/data/temp/kaggle/'

data_store_file = data_dir + '_hotel_search.hdf5'
down_sample_dataset(data_store_file)
train_small = pd.read_hdf(data_store_file, 'train_small')
test_small = pd.read_hdf(data_store_file, 'test_small')

# Classifier 1 - Use the top 5 most common hotel clusters
# Get the ids of the top 5 most common hotel clusters
most_common_clusters = list(train.hotel_cluster.value_counts().head().index)
# For each test sample, we provide 5 predictions i.e. 5 most common clusters
predictions = [most_common_clusters for i in range(len(test_small))]

# Convert target to a list to be used in the mapk method
target = [[l] for l in test_small["hotel_cluster"]]
metrics.mapk(target, predictions, k=5) # k = number of predictions??

# Finding correlations
# The closer it is to 1, the stronger the (linear) correlation
# For this training dataset, there is no strong linear correlation between hotel_cluster and other variables
train.corr()['hotel_cluster']

# Feature generation
# PCA (A very simple and easy-to-understand explanation of dimenionality reduction using PCA can be found here:https://georgemdallas.wordpress.com/2013/10/30/principal-component-analysis-4-dummies-eigenvectors-eigenvalues-and-dimension-reduction/)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
# Tranform columns d1 to d149 to 3 components
dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]]) # output format is numpy ndarray (n-dimensional array)
dest_small = pd.DataFrame(dest_small) # Convert to data frame
dest_small["srch_destination_id"] = destinations["srch_destination_id"]


df = calc_fast_features(train_small)
df.fillna(-1, inplace=True) # Replace NA value with -1

def first_classifier():


def calc_fast_features(df):
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], format='%Y-%m-%d', errors="coerce")
    df["srch_co"] = pd.to_datetime(df["srch_co"], format='%Y-%m-%d', errors="coerce")

    props = {}
    for prop in ["month", "day", "hour", "minute", "dayofweek", "quarter"]:
        props[prop] = getattr(df["date_time"].dt, prop)

    carryover = [p for p in df.columns if p not in ["date_time", "srch_ci", "srch_co"]]
    for prop in carryover:
        props[prop] = df[prop]

    date_props = ["month", "day", "dayofweek", "quarter"]
    for prop in date_props:
        props["ci_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
        props["co_{0}".format(prop)] = getattr(df["srch_co"].dt, prop)
    props["stay_span"] = (df["srch_co"] - df["srch_ci"]).astype('timedelta64[h]')

    ret = pd.DataFrame(props)

    ret = ret.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest")
    ret = ret.drop("srch_destination_iddest", axis=1)
    return ret


def down_sample_dataset(data_store_file):
    destinations = pd.read_csv(data_dir + "destinations.csv")  # Destination of where the hotel search was performed (e.g. Gold Coast, Santuary Cove) and its latent variables
# test = pd.read_csv(data_dir + "test.csv", parse_dates=[2])
train = pd.read_csv(data_dir + "train.csv")

# Check dataset sizes
train.shape
train.columns.values
destinations.columns.values

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

# Down-sample using 10000 users
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