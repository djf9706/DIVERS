import torch

train_batch_size = 1024
test_batch_size = 1024

device = 'cuda'


l2_lambda = 1e-8


coat = 32

yahoo = 96

kuai = 128



# coat

coat_phi = {'hidden_dims': [32, 32, 32],
            'dropout': [0.1, 0.1],
            'is_dropout': True
            }

coat_alpha = {'hidden_dims': [32 * 2, 512, 128, 32, 1],
              'dropout': [], 'is_dropout': False
              }
coat_beta = {'hidden_dims': [32 * 2, 512, 128, 32, 1],
              'dropout': [], 'is_dropout': False
              }




# yahoo  local(3)

yahoo_phi = {'hidden_dims': [96, 96, 96],
            'dropout': [0.1, 0.1],
            'is_dropout': True
            }

yahoo_alpha = {'hidden_dims': [96 * 2, 512, 128, 32, 1],
              'dropout': [], 'is_dropout': False
              }
yahoo_beta = {'hidden_dims': [96 * 2, 512, 128, 32, 1],
              'dropout': [], 'is_dropout': False
              }


# kuai
kuai_phi = {'hidden_dims': [128, 128, 128],
            'dropout': [0.1, 0.1],
            'is_dropout': True
            }


kuai_alpha = {'hidden_dims': [128 * 2, 1, 1, 1, 1],
              'dropout': [], 'is_dropout': False
              }
kuai_beta = {'hidden_dims': [128 * 2, 1, 1, 1, 1],
              'dropout': [], 'is_dropout': False
              }







# sim

sim_phi = {'hidden_dims': [128, 128, 128],
            'dropout': [0.1, 0.1],
            'is_dropout': True
            }

sim_alpha = {'hidden_dims': [128 * 2, 512, 128, 32, 1],
              'dropout': [], 'is_dropout': False
              }
sim_beta = {'hidden_dims': [128 * 2, 512, 128, 32, 1],
              'dropout': [], 'is_dropout': False
              }


