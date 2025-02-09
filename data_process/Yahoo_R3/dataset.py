import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
train_table_raw = pd.read_table("/yahoo-20240107T063527Z-001/yahoo/ydata-ymusic-rating-study-v1_0-train.txt", sep="\t", header=None)
test_table_raw = pd.read_table("/yahoo-20240107T063527Z-001/yahoo/ydata-ymusic-rating-study-v1_0-test.txt", sep="\t", header=None)
user_feature = pd.read_table("ydata-ymusic-rating-study-v1_0-survey-answers.txt", sep="\t", header=None)
n_users_with_feat = user_feature.shape[0]
n_users_with_feat
train_array = train_table_raw.to_numpy()
test_array = test_table_raw.to_numpy()
user_feature_array = user_feature.to_numpy()
train_df = pd.DataFrame(train_array, columns=["user_id", "item_id", "rating"])
random_df = pd.DataFrame(test_array, columns=["user_id", "item_id", "rating"])
train_df
train_df["user_id"] -= 1
train_df["item_id"] -= 1
random_df["user_id"] -= 1
random_df["item_id"] -= 1
train_df = train_df.loc[train_df["user_id"] < n_users_with_feat]
random_df = random_df.loc[random_df["user_id"] < n_users_with_feat]
train_df
train_df.to_csv("train.csv", index=False)
random_df.to_csv("random.csv", index=False)
pd.read_csv("random.csv")
user_feature
user_feat_onehot = pd.get_dummies(user_feature[0])
for i in range(1, 7):
    df = pd.get_dummies(user_feature[i])
    user_feat_onehot = pd.concat([user_feat_onehot, df], axis=1)
user_feat_onehot
user_feat_onehot.to_csv("user_feat_onehot.csv", index=False)

(user_feature - 1).to_csv("user_feat_label.csv", index=False)
