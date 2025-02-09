import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


df_train1 = pd.read_csv("KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv")
df_train2 = pd.read_csv("KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv")

df_test_raw = pd.read_csv("KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv")

df_train_raw = df_train1.append(df_train2, ignore_index=True)

df_train_raw = df_train_raw.sort_values(by="user_id")

train_users = df_train_raw["user_id"].unique()
test_users = df_test_raw["user_id"].unique()

a = df_train_raw["user_id"].value_counts()
valid_user_count = a[a >= 10]
df_train_filter_u = df_train_raw.loc[df_train_raw["user_id"].isin(valid_user_count.index)]
b = df_train_filter_u["video_id"].value_counts()
valid_item_count = b[b >= 10]
df_train_filter_ui = df_train_filter_u.loc[df_train_filter_u["video_id"].isin(valid_item_count.index)]
df_train_filter_ui["user_id"].value_counts()

df_test_raw["user_id"].value_counts()
invalid_test_users = np.setdiff1d(test_users, df_train_filter_ui["user_id"].unique())
df_test_filter_u = df_test_raw.loc[~df_test_raw["user_id"].isin(invalid_test_users)]
df_test_filter_u
np.setdiff1d(df_train_raw["video_id"].unique(), df_test_raw["video_id"].unique())
invalid_test_items = np.setdiff1d(df_test_filter_u["video_id"].unique(), df_train_filter_ui["video_id"].unique())
df_test_filter_ui = df_test_filter_u.loc[~df_test_filter_u["video_id"].isin(invalid_test_items)]
df_test_filter_ui
df_test_filter_ui["user_id"].value_counts()
df_train_filter_ui[df_train_filter_ui["is_click"] == 1]["user_id"].value_counts().describe()
df_test_filter_ui[df_test_filter_ui["is_click"] == 1]["user_id"].value_counts().describe()
df_train_filter_ui[df_train_filter_ui["is_like"] == 1]["user_id"].value_counts().describe()
df_train = df_train_filter_ui.rename(columns={
    "video_id": "item_id",
    "is_click": "rating"
})
df_train = df_train[["user_id", "item_id", "rating"]]
df_random = df_test_filter_ui.rename(columns={
    "video_id": "item_id",
    "is_click": "rating"
})
df_random = df_random[["user_id", "item_id", "rating"]]
unique_users = df_train["user_id"].unique()
unique_items = df_train["item_id"].unique()
user_map = dict(zip(unique_users, np.arange(unique_users.shape[0])))
item_map = dict(zip(unique_items, np.arange(unique_items.shape[0])))
df_train["user_id"] = df_train["user_id"].map(lambda x: user_map[x])
df_random["user_id"] = df_random["user_id"].map(lambda x: user_map[x])
df_train["item_id"] = df_train["item_id"].map(lambda x: item_map[x])
df_random["item_id"] = df_random["item_id"].map(lambda x: item_map[x])
df_train["user_id"].unique()
df_train["item_id"].unique()
user_feat_df = pd.read_csv("KuaiRand-Pure/data/user_features_pure.csv")
user_feat_df = user_feat_df.loc[user_feat_df["user_id"].isin(user_map.keys())]
user_feat_df["user_id"] = user_feat_df["user_id"].map(lambda x: user_map[x])
user_feat_df
user_feat_df.drop(
    columns=["user_id", "is_lowactive_period", "follow_user_num", "fans_user_num", "friend_user_num", "register_days"],
    inplace=True)
for name in ["user_active_degree", "follow_user_num_range", "fans_user_num_range", "friend_user_num_range",
             "register_days_range"]:
    user_feat_df[name] = pd.Categorical(user_feat_df[name]).codes
user_feat_df.loc[user_feat_df["is_live_streamer"] == -124, "is_live_streamer"] = 0
user_feat_df
for col in user_feat_df.columns:
    if user_feat_df[col].isna().sum() > 0:
        user_feat_df.drop(columns=[col], inplace=True)
user_feat_df
user_feat_df.drop(columns=["onehot_feat3", "onehot_feat5", "onehot_feat7", "onehot_feat8"], inplace=True)
feature_dims = []
for col in user_feat_df.columns:
    user_feat_df[col] = user_feat_df[col].astype(int)
    feature_dims.append(user_feat_df[col].max() + 1)
print(feature_dims)
user_feat_df
user_feat_df.to_csv("user_feat_label.csv", index=None)
for col in user_feat_df.columns:
    df = pd.get_dummies(user_feat_df[col], prefix=col)
    user_feat_df.drop(columns=[col], inplace=True)
    user_feat_df = user_feat_df.join(df)
user_feat_df.to_csv("user_feat_onehot.csv", index=None)
df_train.to_csv("train.csv", index=False)
df_random.to_csv("random.csv", index=False)
user_feat_df
df_train
density = df_train["user_id"].value_counts() / (df_train["item_id"].max() + 1)
density
density.mean()