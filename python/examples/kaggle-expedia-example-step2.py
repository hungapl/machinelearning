# Exploring Kaggle Expedia Competition dataset based on https://www.dataquest.io/blog/kaggle-tutorial/
# Classifier comparison http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

import pandas as pd

def calc_fast_features(df):
    # Extract time fields as features
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], format='%Y-%m-%d')
    df["srch_co"] = pd.to_datetime(df["srch_co"], format='%Y-%m-%d')
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

    # Infer stay length from check-in and check-out date
    props["stay_span"] = (df["srch_co"] - df["srch_ci"]).astype('timedelta64[h]')

    ret = pd.DataFrame(props)

    # Add destination variable columns to training dataset
    ret = ret.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest")
    ret = ret.drop("srch_destination_iddest", axis=1)
    return ret


data_dir = '/home/hungap/data/temp/kaggle/'
data_store_file = data_dir + '_hotel_search.hdf5'
destinations = pd.read_csv(data_dir + "destinations.csv")  # Destination of where the hotel search was performed (e.g. Gold Coast, Santuary Cove) and its latent variables
train_small = pd.read_hdf(data_store_file, 'train_small')
test_small = pd.read_hdf(data_store_file, 'test_small')

# Feature generation
# PCA (A very simple and easy-to-understand explanation of dimenionality reduction using PCA can be found here:https://georgemdallas.wordpress.com/2013/10/30/principal-component-analysis-4-dummies-eigenvectors-eigenvalues-and-dimension-reduction/)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
# Tranform columns d1 to d149 to 3 components
dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]]) # output format is numpy ndarray (n-dimensional array)
dest_small = pd.DataFrame(dest_small) # Convert to data frame
dest_small["srch_destination_id"] = destinations["srch_destination_id"]

# Plotting
# from bokeh.plotting import figure, show
# import itertools
#
# p = figure(plot_width=1200, plot_height=900)
# colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
# colors = [colormap[x] for x in flowers['species']]
# p.circle(flowers["petal_length"], flowers["petal_width"],
#          color=colors, fill_alpha=0.2, size=10)
#
# labels = ['logistic regression', 'random forest', 'naive bayes', 'ensemble']
# for clf, lab, grd in zip([clf1, clf2, clf3, eclf],
#                          labels,
#                          itertools.product([0, 1], repeat=2)):
#
#     clf.fit(x, y)
#     ax = plt.subplot(gs[grd[0], grd[1]])
#     fig = plot_decision_regions(x=x, y=y, clf=clf)
#     plt.title(lab)

import random

from sklearn.model_selection import train_test_split, cross_val_score
train_feature = calc_fast_features(train_small)
train_feature.fillna(-1, inplace=True) # Replace NA value with -1, replace views as well when inplace=T
predictors = [c for c in train_feature.columns if c not in ["hotel_cluster"]]

# Try binary classifier for most common hotel cluster
top_cluster = train_feature.hotel_cluster.value_counts().head(1).index
train_top = train_feature[train_feature.hotel_cluster.isin(top_cluster)]
is_not_top = train_feature[~train_feature.hotel_cluster.isin(top_cluster)]
is_not_top = is_not_top.iloc[random.choices(range(1, len(is_not_top)), k=len(train_top))]
is_not_top['hotel_cluster'] = 0
train_top = train_top.append(is_not_top)
X_train, X_test, y_train, y_test = train_test_split(train_top[predictors], train_top['hotel_cluster'])


#######
## Classifier 2: Random Forest
#######
#train_feature.corr()['hotel_cluster']
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1).fit(X_train, y_train)
#cross_val_score(clf, X_test, y_test, cv=3)


######
## Classifier 3: SVM multi-class classification
######
# from sklearn.preprocessing import MinMaxScaler
# scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
# X_train = scaling.transform(X_train)
# X_test = scaling.transform(X_test)

from sklearn import svm
clf = svm.SVC().fit(X_train, y_train)

######
## Classifier 3 - Group similar users together
######


