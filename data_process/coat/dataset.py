import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
train_table_raw = pd.read_table("train.ascii", sep=" ", header=None)
test_table_raw = pd.read_table("test.ascii", sep=" ", header=None)
user_feature = pd.read_table("user_item_features/user_features.ascii", sep=" ", header=None)
item_feature = pd.read_table("user_item_features/item_features.ascii", sep=" ", header=None)
train_array = train_table_raw.to_numpy()
test_array = test_table_raw.to_numpy()
user_feature_array = user_feature.to_numpy()
item_feature_array = item_feature.to_numpy()
def to_df(data_matrix):
    user_id, item_id = data_matrix.nonzero()
    rating = data_matrix[user_id, item_id]
    data = np.concatenate(
        (user_id.reshape(-1, 1),
         item_id.reshape(-1, 1),
         rating.reshape(-1, 1)),
        axis=1)
    return pd.DataFrame(data, columns=["user_id", "item_id", "rating"])
train_df = to_df(train_array)
random_df = to_df(test_array)
train_df.to_csv("train.csv", index=False)
random_df.to_csv("random.csv", index=False)
pd.read_csv("train.csv")
user_gender_id = np.nonzero(user_feature_array[:, 0:2])[1]
age_id = np.nonzero(user_feature_array[:, 2:8])[1]
location_id = np.nonzero(user_feature_array[:, 8:11])[1]
fashioninterest_id = np.nonzero(user_feature_array[:, 11:14])[1]
user_feature = np.concatenate(
    (
        user_gender_id.reshape(-1, 1),
        age_id.reshape(-1, 1),
        location_id.reshape(-1, 1),
        fashioninterest_id.reshape(-1, 1)
    ),
    axis=1
)
user_feature
pd.DataFrame(user_feature_array).to_csv("user_feat_onehot.csv", index=False)
pd.DataFrame(user_feature).to_csv("user_feat_label.csv", index=False)