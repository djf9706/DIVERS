

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import torch.nn as nn

from cal_mcc import mean_corr_coef

np.random.seed(1234)
torch.manual_seed(1234)

# Parameters
n_labels = 5
sample_size = 10000
z_dim = 2
x_dim = 1000
embedding_dim = 16
random_item_size = 15
sparse_ratio = 0.1
rating_noise_ratio = 10.0
treatment_noise_ratio = 0.0
confounding_effect_rating = 2.0

name_suffix = "_sr_{}_cr_{}_nr_10.0_tr_{}".format(sparse_ratio, confounding_effect_rating, treatment_noise_ratio)



def gen_confounder(sample_size, label_size, confounder_dim, noise_level=0.1):

    mu_true = np.random.uniform(-12, 12, [confounder_dim, label_size])
    var_true = np.random.uniform(1, 5, [confounder_dim, label_size])


    mu_true_1 = np.random.normal(-10, 10, [confounder_dim, label_size])
    var_true_1 = np.random.uniform(1, 10, [confounder_dim, label_size])


    w_true = np.random.choice(np.arange(label_size), sample_size, p=[0.08, 0.2, 0.2, 0.32, 0.2])


    z_true = np.vstack([
        np.random.normal(mu_true[i][w_true], np.sqrt(var_true[i][w_true])) for i in range(confounder_dim)
    ]).T


    z_i = np.vstack([
        np.random.normal(mu_true_1[i][w_true], np.sqrt(var_true_1[i][w_true])) for i in range(confounder_dim)
    ]).T



    z_true = z_true + 0.1*z_i


    z_true += noise_level * np.random.randn(*z_true.shape)

    return w_true, z_true, mu_true, var_true






# Generate treatment (exposure) a_ui
def gen_treatment(treatment_dim, confounder, emb_z, sparse_ratio, treatment_noise_ratio=0, gamma=2.0):
    W = 0.3 * torch.rand((confounder.shape[1], confounder.shape[1]))  # Matrix M
    x_prob = nn.LeakyReLU(0.2)(confounder @ W @ emb_z.T)  # LeakyReLU(z M e_z)

    noise = (torch.randn_like(x_prob)) * (treatment_noise_ratio) # γ * ε
    x_prob += gamma * noise  # Add noise

    x_prob = torch.sigmoid(x_prob) * sparse_ratio  # Sigmoid and sparsity control (α)
    return x_prob, torch.bernoulli(x_prob)  # Bernoulli sampling for exposure




# Generate embeddings for users, items, and confounders
def gen_gaussian_embedding(size, embedding_dim, add_bias=False):
    emb = torch.randn((size, embedding_dim)) * 5
    if add_bias:
        bias = torch.randint(4, (size, 1))
        emb += bias
    return emb


def gen_uniform_embedding(size, embedding_dim):
    emb = torch.rand((size, embedding_dim))
    return emb


# Generate confounder data
w_true, z_true, mu_true, var_true = gen_confounder(sample_size, n_labels, z_dim)
z_true = torch.tensor(z_true, dtype=torch.float)


def gen_log_normal_embedding(size, embedding_dim):
    embedding = torch.exp(torch.randn((size, embedding_dim)) * 0.5)  # 使用对数正态分布
    return embedding

def gen_mixed_embeddingi(sample_size, embedding_dim, normal_ratio=0.9):

    normal_size = int(sample_size * normal_ratio)


    normal_embedding = torch.randn((normal_size, embedding_dim)) * 5

    bias = torch.randint(4, (normal_size, 1))
    normal_embedding += bias



    long_size = sample_size - normal_size
    long_embedding = torch.exp(torch.randn((long_size, embedding_dim)) * 0.5)


    embedding = torch.cat([normal_embedding, long_embedding], dim=0)


    indices = torch.randperm(sample_size)
    mixed_embedding = embedding[indices]

    return mixed_embedding

def gen_mixed_embedding(sample_size, embedding_dim, normal_ratio=0.9):

    normal_size = int(sample_size * normal_ratio)
    normal_embedding = torch.randn((normal_size, embedding_dim))  # 正态分布 N(0,1)


    uniform_size = sample_size - normal_size
    uniform_embedding = torch.rand((uniform_size, embedding_dim))  # 均匀分布 U(0,1)


    embedding = torch.cat([normal_embedding, uniform_embedding], dim=0)


    indices = torch.randperm(sample_size)
    mixed_embedding = embedding[indices]

    return mixed_embedding



# # Generate user and item embeddings
emb_u = gen_uniform_embedding(sample_size, embedding_dim)
emb_i = gen_gaussian_embedding(x_dim, embedding_dim, add_bias=True)
emb_z = gen_uniform_embedding(x_dim, z_dim)

# Generate treatment data
x_prob, x_obs = gen_treatment(x_dim, z_true, emb_z, sparse_ratio, treatment_noise_ratio)



# Generate ratings with confounding effect and noise
exp_effect = emb_u @ emb_i.T
confounder_effect = z_true @ emb_z.T * confounding_effect_rating

# Added smaller rating noise
noise = torch.randn((sample_size, x_dim)) * rating_noise_ratio
mf_res = exp_effect + confounder_effect + noise

# Nonlinear transformation and normalization
soft_mf_res = torch.pow(
    (mf_res - torch.quantile(mf_res, 0.05)) / (torch.quantile(mf_res, 0.95) - torch.quantile(mf_res, 0.05)), 1)

# Transform ratings to integer values in range [1, 5]
rating_matrix = torch.ceil(soft_mf_res * 5)
rating_matrix = torch.clamp(rating_matrix, min=1, max=5)





# Plot effects
plt.hist(exp_effect.abs().mean(1))
plt.hist(confounder_effect.abs().mean(1))
plt.hist(noise.abs().mean(1))

# Sample random interactions for the dataset
uids, iids = x_obs.nonzero(as_tuple=True)
ratings = rating_matrix[uids, iids]

# Random sampling of items
random_iids_list = [torch.randperm(x_dim)[:random_item_size] for _ in range(sample_size)]
random_iids = torch.cat(random_iids_list)
random_uids = torch.arange(0, sample_size).view(-1, 1).repeat(1, random_item_size).view(-1)
random_ratings = rating_matrix[random_uids, random_iids]


# Function to save data to CSV
def save_csv(uids, iids, ratings, name):
    df = pd.DataFrame(
        data={"user_id": uids.numpy(), "item_id": iids.numpy(), "rating": ratings.numpy()}
    )
    df.to_csv(name, sep=",", index=None)
    return df


# Save the datasets
df_train = save_csv(uids, iids, ratings, f"train{name_suffix}.csv")
df_random = save_csv(random_uids, random_iids, random_ratings, f"random{name_suffix}.csv")

# Save user feature labels
user_feat_onehot = pd.get_dummies(w_true)
pd.Series(w_true).to_csv("user_feat_label.csv", index=None)
user_feat_onehot.to_csv("user_feat_onehot.csv", index=None)



plt.figure(figsize=(6, 6))

plt.scatter(z_true.T[0], z_true.T[1], c=w_true, s=1)

plt.show()


ivae_z_mean = torch.load("../simiv/iv/mean.pt")
ivae_z_mean = ivae_z_mean.to('cpu')

plt.figure(figsize=(6, 6))
# ax1 = plt.subplot(2, 2, 1)
# ax1.set_title("True 2-dim latent")
# plt.title("")
# plt.scatter(z_true.T[0], z_true.T[1], c=w_true, s=1)
# ax2 = plt.subplot(2, 2, 2)
# ax2.set_title("VAE")
# plt.scatter(vae_z_mean.T[0].detach().numpy(), vae_z_mean.T[1].detach().numpy(), c=w_true, s=1)
# ax3 = plt.subplot(2, 2, 3)
# ax3.set_title("iVAE")
plt.scatter(ivae_z_mean.T[0].detach().numpy(), ivae_z_mean.T[1].detach().numpy(), c=w_true, s=1)
# ax3 = plt.subplot(2, 2, 4)
# ax3.set_title("iVAE_IV")
# plt.scatter(ivae_iv_z_mean.T[0].detach().numpy(), ivae_iv_z_mean.T[1].detach().numpy(), c=w_true, s=1)


plt.show()


print(mean_corr_coef(z_true, ivae_z_mean).item())