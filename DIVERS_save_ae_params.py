from models.divivae import *
from argparser import *
from utils import *
from pathlib import Path
def save_ae_params(data_params, ae_suffix, sr=None, tr=None, seed=1234, source_dir="tuned_models/"):
    df = pd.read_csv(data_params["train_path"])
    n_users = df["user_id"].max() + 1
    n_items = df["item_id"].max() + 1
    full_matrix = df_to_csr(df, shape=(n_users, n_items))
    i = df["item_id"].to_numpy()

    length = i.shape[0]
    chunk_size = length // n_users


    new_length = chunk_size * n_users


    i = i[:new_length].reshape(-1, chunk_size)


    i = torch.tensor(i, dtype=torch.float)
    full_data = torch.tensor(full_matrix.toarray() > 0, dtype=torch.float)
    user_feat = torch.tensor(pd.read_csv(data_params["user_feature_path"]).to_numpy(), dtype=torch.float)

    query_embedding_matrix, _ = load_pretrained_embedding()


    query_embedding_layer1 = nn.Embedding.from_pretrained(query_embedding_matrix, freeze=True).to("cuda")

    corresponding_query_matrix = load_corresponding_query()

    corresponding_query_embedding = nn.Embedding.from_pretrained(corresponding_query_matrix, freeze=True).to('cuda')

    i1 = i.to(torch.long).to("cuda")

    i_embedding = query_embedding_layer1(i1)


    i_embedding = i_embedding.view(i1.size(0), -1)

    cor_query_embedding = corresponding_query_embedding(i1).reshape(i1.shape[0], -1)



    treatment_model = MLPIV(input_dim=cor_query_embedding.shape[1], b="",
                            output_dim=i_embedding.shape[1]).to("cuda")

    pytorch_deep_iv = PyTorchDeepIV(
        n_components=10,
        m=treatment_model,  # Pass the model instance
        h=1,
        n_samples=1,
        use_upper_bound_loss=False,
        n_gradient_samples=1,
        optimizer='adam',
        first_stage_options={'epochs': 50, 'lr': 1e-3},
    )

    pytorch_deep_iv.fit(T=i_embedding, Z=cor_query_embedding)

    treatment_model.eval()


    with torch.no_grad():

        treatment_pred = treatment_model(cor_query_embedding)


    model_type = "ivae" if "ivae" in ae_suffix else "vae"

    if data_params["name"] == "sim":
        sim_suffix = "_sr_{}_cr_2.0_nr_10.0_tr_{}".format(sr, tr)
        data_params["train_path"] = dir_prefix + "/data_process/simulation/train{}.csv".format(sim_suffix)
        model_path = "{}sr_{}_tr_{}/{}_{}".format(source_dir, sr, tr, data_params["name"], ae_suffix)
        dir_name = "data_process/{}_{}/sr_{}_tr_{}/".format(data_params["name"], model_type, sr, tr)
    else:
        model_path = "{}{}_{}".format(source_dir, data_params["name"], ae_suffix)
        dir_name = "data_process/{}_{}/".format(data_params["name"], model_type)
    Path(dir_name).mkdir(parents=True, exist_ok=True)

    save_name_suffix = ""
    if "seed" in ae_suffix:
        save_name_suffix = "_" + ae_suffix.split("_")[3]
    if model_type == "ivae":
        prior_mean = True if data_params["name"] == "sim" else False
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        print(state_dict.keys())
        hidden_dim = state_dict['mean_z.linear_nets.fc_0.weight'].shape[0]
        latent_dim = state_dict['mean_z.linear_nets.fc_2.weight'].shape[0]
        ivae_model = iVAE(input_dim=n_items, auxiliary_dim=user_feat.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim,
                          n_layers=3, prior_mean=prior_mean)
        ivae_model.load_state_dict(state_dict)
        x_hat, mean, log_var, prior_mean, prior_log_var = ivae_model(full_data, user_feat, treatment_pred)
        torch.save(mean.detach(), dir_name + "mean{}.pt".format(save_name_suffix))
        torch.save(log_var.detach().exp().sqrt(), dir_name + "std{}.pt".format(save_name_suffix))
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        hidden_dim = state_dict['mean_z.linear_nets.fc_0.weight'].shape[0]
        latent_dim = state_dict['mean_z.linear_nets.fc_2.weight'].shape[0]
        vae_model = VAE(input_dim=n_items, latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=3)
        vae_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        x_hat, mean, log_var = vae_model(full_data)
        torch.save(mean.detach(), dir_name + "mean{}.pt".format(save_name_suffix))
        torch.save(log_var.detach().exp().sqrt(), dir_name + "std{}.pt".format(save_name_suffix))

save_ae_params(coat_params,"val_ivae_exposure_best.pt")