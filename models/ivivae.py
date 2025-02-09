import torch
import torch.nn as nn

import config
from data.loader import load_pretrained_embedding, load_corresponding_query


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activations, device="cuda"):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.device = device

        self.linear_nets = nn.Sequential()
        prev_dim = input_dim
        for i, (hidden_dim, activation) in enumerate(zip(hidden_dims, activations)):
            self.linear_nets.add_module("fc_{}".format(i), nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
            if activation == "relu":
                self.linear_nets.add_module("act_{}".format(i), nn.ReLU())
            elif activation == "lrelu":
                self.linear_nets.add_module("act_{}".format(i), nn.LeakyReLU(0.2))
            elif activation == "sigmoid":
                self.linear_nets.add_module("act_{}".format(i), nn.Sigmoid())
            elif activation == "softmax":
                self.linear_nets.add_module("act_{}".format(i), nn.Softmax(dim=1))
            elif activation == "tanh":
                self.linear_nets.add_module("act_{}".format(i), nn.Tanh())

        self.to(self.device)

    def forward(self, x):
        x = x.to('cuda')
        return self.linear_nets(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        h = self.LeakyReLU(self.FC_input(x))
        h = self.LeakyReLU(self.FC_input2(h))
        mean = self.FC_mean(h)
        log_var = self.FC_var(h)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, n_layers=3, activation="lrelu", out_activation=None,
                 device="cpu"):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        # encoder
        self.mean_z = MLP(input_dim=input_dim,
                          hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation],
                          device=device)
        self.log_var_z = MLP(input_dim=input_dim,
                             hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation],
                             device=device)

        # decoder
        self.decoder = MLP(input_dim=latent_dim,
                           hidden_dims=[hidden_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + [out_activation],
                           device=device)


    def encode(self, x):
        mean = self.mean_z(x)
        log_var = self.log_var_z(x)
        return mean, log_var

    def decode(self, x):
        return self.decoder(x)

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat, mean, log_var

    def reconstruct(self, x, sample=False):
        mean, log_var = self.encode(x)
        z = mean
        if sample:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat


class iVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, auxiliary_dim, hidden_dim, n_layers=3, activation="lrelu",
                 out_activation=None, device="cuda", prior_mean=False, dropout=0.):
        super(iVAE, self).__init__()
        self.latent_dim = latent_dim

        self.device = device
        self.to(self.device)

        # prior params
        self.prior_mean = prior_mean
        if self.prior_mean:
            self.prior_mean_z = MLP(input_dim=auxiliary_dim,
                                    hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                                    activations=[activation] * (n_layers - 1) + [out_activation],
                                    device=device)
        self.prior_log_var_z = MLP(input_dim=auxiliary_dim,
                                   hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                                   activations=[activation] * (n_layers - 1) + [out_activation],
                                   device=device)

        # encoder params
        self.mean_z = MLP(input_dim=input_dim + auxiliary_dim,
                          hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation],
                          device=device)
        self.log_var_z = MLP(input_dim=input_dim + auxiliary_dim,
                             hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation],
                             device=device)

        # decoder params
        self.decoder = MLP(input_dim=latent_dim,
                           hidden_dims=[hidden_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + [out_activation],
                           device=device)


        query_embedding_matrix, photo_embedding_matrix = load_pretrained_embedding()

        self.query_embedding_layer1 = nn.Embedding.from_pretrained(query_embedding_matrix, freeze=True).to(self.device)

        corresponding_query_matrix, cor_query_pinv = load_corresponding_query()
        self.corresponding_query_embedding = nn.Embedding.from_pretrained(corresponding_query_matrix, freeze=True).to('cuda')
        self.cor_query_pinv = nn.Embedding.from_pretrained(cor_query_pinv, freeze=True).to('cuda')

        self.regularization_weight = []

        from .modules.phi1 import MLP1

        #change dataset setting
        self.phi = MLP1(config.coat_phi)

        self.add_regularization_weight(self.phi.parameters(), l2=config.l2_lambda)


        self.alpha = MLP1(config.coat_alpha)
        self.add_regularization_weight(self.alpha.parameters(), l2=config.l2_lambda)
        self.beta = MLP1(config.coat_beta)
        self.add_regularization_weight(self.beta.parameters(), l2=config.l2_lambda)

        self.drop = nn.Dropout(dropout)
        self.mean = nn.Parameter(torch.FloatTensor([0]), False)

        self.prob_sigmoid = nn.Sigmoid()



    def encode(self, x, w):
        x = x.to('cuda')
        w = w.to('cuda')
        xw = torch.cat((x, w), 1)
        mean = self.mean_z(xw)
        log_var = self.log_var_z(xw)
        return mean, log_var

    def decode(self, x):
        return self.decoder(x)

    def prior(self, w):
        log_var_z = self.prior_log_var_z(w)
        if self.prior_mean:
            mean_z = self.prior_mean_z(w)
        else:
            mean_z = torch.zeros_like(log_var_z).to(self.device)
        return mean_z, log_var_z

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def iv(self, photo, query, query_pinv):

        self.alpha = self.alpha.to('cuda')
        self.beta = self.beta.to('cuda')

        query_origin = query.flatten(start_dim=2)

        alpha = self.prob_sigmoid(self.alpha(torch.cat([photo, query_origin], dim=-1)))

        beta = self.prob_sigmoid(self.beta(torch.cat([photo, query_origin], dim=-1)))
        photo = photo.unsqueeze(dim=-1)

        t_1 = torch.matmul(query, torch.matmul(query_pinv, photo))

        t_2 = photo - t_1


        q_t = torch.matmul(t_2, alpha.unsqueeze(dim=-1)) + torch.matmul(t_1, beta.unsqueeze(dim=-1))



        return q_t.squeeze_(dim=-1)

    def forward(self, x, w, i):



        i1 = x.shape[1]

        i2 = i.shape[1]
        # i = i.to(torch.long).to('cuda')
        i = i.to(torch.long).to(self.device)


        i_embedding = self.query_embedding_layer1(i)
        query_num_per_photo = 1
        cor_query_embedding = self.corresponding_query_embedding(i).reshape(i.shape[0], i2, config.coat,
                                                                            query_num_per_photo)
        cor_query_pinv = self.cor_query_pinv(i).reshape(i.shape[0], i2, query_num_per_photo, config.coat)

        self.phi = self.phi.to('cuda')
        i_embedding = self.phi(i_embedding)

        i = self.iv(i_embedding, cor_query_embedding, cor_query_pinv)



        i = i.view(x.size(0), -1)
        fc_layer = nn.Linear(i.shape[1], i1).to('cuda')
        i = fc_layer(i)




        x = x.to('cuda')
        i = i.to('cuda')

        x1 = x
        x = x + i


        prior_log_var = self.prior_log_var_z(w)

        if self.prior_mean:
            prior_mean = self.prior_mean_z(w)
        else:
            prior_mean = torch.zeros_like(prior_log_var).to(self.device)


        mean, log_var = self.encode(x, w)
        mean1, log_var1 = self.encode(x1, w)

        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        z1 = self.reparameterization(mean1, torch.exp(0.5 * log_var1))

        z = 0.9*z + 0.9*z1


        x_hat = self.decode(z)
        return x_hat, mean, log_var, prior_mean, prior_log_var

    def reconstruct(self, x, w, sample=False):
        mean, log_var = self.encode(x, w)
        z = mean
        if sample:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat
