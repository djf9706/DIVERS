from models.divivae import *
from torch import nn
from torch.optim import Adam, SGD
from matplotlib import pyplot as plt
from utils import *
from torch.utils.data import Dataset, DataLoader
from argparser import *
from ray.air import session
from tune_script import *


def loss_function(x, x_hat, mean, log_var, prior_mean, log_prior_var, anneal=1., mask=None):
    if mask is None:
        reproduction_loss = torch.mean(
            torch.sum(nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction="none"), dim=1))
    else:
        entropy = nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction="none")
        make_sure = x * mask
        assert make_sure.sum() == 0
        valid_mask = torch.logical_not(torch.logical_and(torch.logical_not(x), mask))
        filter_entropy = entropy * valid_mask
        reproduction_loss = torch.mean(
            filter_entropy.sum(dim=1)
        )
    kld = -0.5 * torch.mean(
        torch.sum(
            1 + log_var - log_prior_var - ((mean - prior_mean).pow(2) + log_var.exp()) / log_prior_var.exp(),
            dim=1)
    )


    return reproduction_loss + kld * anneal


class TrainingDataset(Dataset):
    def __init__(self, data, feature, tm, mask=None):
        self.data = data
        self.feature = feature
        self.tm = tm
        self.mask = mask

    def __getitem__(self, index):
        if self.mask is None:
            return self.data[index], self.feature[index], self.tm[index]
        else:
            return self.data[index], self.feature[index], self.tm[index], self.mask[index]

    def __len__(self):
        return len(self.data)


def train_eval(config):
    params = config["data_params"]

    train_ratio = params["train_ratio"]
    train_matrix, val_matrix, train_user_index, test_user_index, tm, vm = construct_vae_dataset(params["train_path"],
                                                                                        train_ratio=train_ratio,
                                                                                        split_test=False, )
    user_feat = pd.read_csv(params["user_feature_path"]).to_numpy()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    train_data = torch.tensor(train_matrix > 0, dtype=torch.float).to(device)
    val_data = torch.tensor(val_matrix > 0, dtype=torch.float).to(device)
    user_feat = torch.tensor(user_feat, dtype=torch.float).to(device)

    tm = torch.tensor(tm, dtype=torch.float).to(device)
    vm = torch.tensor(vm, dtype=torch.float).to(device)

    n_users = train_matrix.shape[0] + val_matrix.shape[0]
    n_items = train_matrix.shape[1]

    train_mask = None
    val_mask = None

    train_dataset = TrainingDataset(data=train_data, feature=user_feat[train_user_index], tm=tm, mask=train_mask)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    hidden_dim = config["hidden_dim"]
    latent_dim = config["latent_dim"]
    user_feat_dim = user_feat.shape[1]

    lr = config["lr"]
    epochs = config["epochs"]

    seed_everything(config["seed"])

    na = config["data_params"]["name"]

    c = config["c"]

    prior_mean = True if params["name"] == "sim" else False
    model = iVAE(input_dim=n_items,
                 auxiliary_dim=user_feat_dim,
                 latent_dim=latent_dim,
                 hidden_dim=hidden_dim,
                 n_layers=config["n_layers"],
                 device=device, prior_mean=prior_mean)
    print(model)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=config["weight_decay"])

    model.train()


    query_embedding_matrix, _ = load_pretrained_embedding()

    query_embedding_layer1 = nn.Embedding.from_pretrained(query_embedding_matrix, freeze=True).to("cuda")

    corresponding_query_matrix = load_corresponding_query()

    corresponding_query_embedding = nn.Embedding.from_pretrained(corresponding_query_matrix, freeze=True).to('cuda')

    vm1 = vm.to(torch.long).to("cuda")

    vm_embedding = query_embedding_layer1(vm1)


    vm_embedding = vm_embedding.view(vm1.size(0), -1)

    cor_query_embedding = corresponding_query_embedding(vm1).reshape(vm1.shape[0], -1)


    treatment_model1 = MLPIV(input_dim=cor_query_embedding.shape[1], b=na, output_dim=vm_embedding.shape[1]).to("cuda")

    pytorch_deep_iv = PyTorchDeepIV(
        n_components=10,
        m=treatment_model1,  # Pass the model instance
        h=1,
        n_samples=1,
        use_upper_bound_loss=False,
        n_gradient_samples=1,
        optimizer='adam',
        first_stage_options={'epochs': c, 'lr': 1e-3},
    )

    pytorch_deep_iv.fit(T=vm_embedding, Z=cor_query_embedding)

    treatment_model1.eval()

    with torch.no_grad():

        treatment_pred1 = treatment_model1(cor_query_embedding)


    best_val_loss = np.inf
    best_training_loss = np.inf
    val_loss_list = []
    training_loss_list = []
    patience_counter = 0
    patience = config["patience"]

    use_anneal = config["anneal"]
    anneal_max = config["beta_max"]
    anneal_count = 0

    total_batches = int(epochs * train_data.shape[0] / config["batch_size"])
    anneal_max_count = int(0.2 * total_batches / anneal_max)



    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_len = 0


        for x in train_dataloader:
            x, w, i = x
            optimizer.zero_grad()

            i1 = i.to(torch.long).to("cuda")

            i_embedding = query_embedding_layer1(i1)

            i_embedding = i_embedding.view(x.size(0), -1)

            cor_query_embedding = corresponding_query_embedding(i1).reshape(i1.shape[0], -1)


            if (na == "coat"):
                seed_everything(config["seed"])

            treatment_model = MLPIV(input_dim=cor_query_embedding.shape[1], b=na,
                                    output_dim=i_embedding.shape[1]).to("cuda")

            pytorch_deep_iv = PyTorchDeepIV(
                n_components=10,
                m=treatment_model,  # Pass the model instance
                h=1,
                n_samples=1,
                use_upper_bound_loss=False,
                n_gradient_samples=1,
                optimizer='adam',
                first_stage_options={'epochs': c, 'lr': 1e-3},
            )

            pytorch_deep_iv.fit(T=i_embedding, Z=cor_query_embedding)

            treatment_model.eval()


            with torch.no_grad():

                treatment_pred = treatment_model(cor_query_embedding)

            x_hat, mean, log_var, prior_mean, prior_log_var = model(x, w, treatment_pred)

            if use_anneal:
                anneal = min(anneal_max, 1. * anneal_count / anneal_max_count)
            else:
                anneal = anneal_max

            l2_reg = torch.tensor([0]).to(device)
            for param in model.parameters():
                l2_reg = l2_reg + torch.norm(param)
            loss = loss_function(x, x_hat, mean, log_var, prior_mean, prior_log_var, anneal, None) \
                   + l2_reg * config["l2_penalty"]
            loss.backward()
            optimizer.step()
            anneal_count += 1
            total_loss += loss.item() * len(x)
            total_len += len(x)

        predict_test_x, mean_val, log_var_val, prior_mean_val, prior_log_var_val = model(val_data,
                                                                                         user_feat[test_user_index], treatment_pred1)
        val_loss = loss_function(val_data, predict_test_x, mean_val, log_var_val, prior_mean_val,
                                 prior_log_var_val, anneal=anneal_max, mask=None).detach().item()
        training_loss = total_loss / total_len


        patience_counter += 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "{}_{}_ivae_exposure_best.pt".format(params["name"], "val"))
            patience_counter = 0
        if training_loss < best_training_loss:
            best_training_loss = training_loss
            torch.save(model.state_dict(), "{}_{}_ivae_exposure_best.pt".format(params["name"], "train"))
        if patience_counter >= patience:
            if config["show_log"]:
                print("reach max patience {}, current epoch {}".format(patience, epoch))
            break

        if config["show_log"]:
            val_loss_list.append(val_loss)
            training_loss_list.append(training_loss)
            print("Epoch {}, Training Loss = {}, Val Loss = {} ".format(epoch, training_loss, val_loss))

    if config["tune"]:
        session.report({
            "training_loss": best_training_loss,
            "val_loss": best_val_loss
        })

    if config["show_log"]:
        plt.plot(val_loss_list, label="Val Loss")
        plt.plot(training_loss_list, label="Training Loss")
        plt.title("iVAE")
        plt.legend()
        plt.show()
        print("Best Training loss = {}, Val loss = {}".format(best_training_loss, best_val_loss))


if __name__ == '__main__':
    args = parse_args()
    print(args.data_params["name"])
    if args.tune:
        config = {
            "tune": True,
            "show_log": False,
            "patience": args.patience,
            "anneal": True,
            "batch_size": args.data_params["batch_size"],
            "lr": tune.grid_search([1e-2, 1e-3, 1e-4]),
            "epochs": 2000,
            "latent_dim": 2 if args.data_params["name"] == "sim" else tune.grid_search([16, 32]),
            "hidden_dim": tune.grid_search([32, 64, 96]),
            "weight_decay": tune.grid_search([1e-5, 1e-6]),
            "l2_penalty": tune.grid_search([0, 0.01, 0.05, 0.1]),
            "n_layers": 3,
            "beta_max": args.data_params["beta_max"],
            "data_params": args.data_params,
            "seed": args.seed,
            "c": tune.grid_search([50, 80, 100])
        }

        res_name = "ivae"
        if args.data_params["name"] == "sim":
            res_name = res_name + args.sim_suffix
        tune_param_exposure(train_eval, config, args, res_name)
    else:

        sample_config = {
            "tune": False,
            "lr": 0.01,
            "epochs": 2000,
            "latent_dim": 16,
            "hidden_dim": 32,
            "beta_max": args.data_params["beta_max"],
            "l2_penalty": 0.1,
            "anneal": True,
            "data_params": args.data_params,
            "patience": 100,
            "weight_decay": 1e-5,
            "batch_size": args.data_params["batch_size"],
            "show_log": True,
            "n_layers": 3,
            "seed": 1234,
            "c": 50

        }

        train_eval(sample_config)
