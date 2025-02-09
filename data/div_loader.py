import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.dataset import IterableDataset


    
def load_pretrained_embedding(root_path='/data/iv/new_data/'):



    # root_path = 'data/IVERS/coat_ivae/'
    # root_path = 'data/IVERS/yahoo_ivae/'
    # root_path = 'data/IVERS/kuai_ivae/'
    # root_path = 'data/IVERS/sim_ivae/'


    root_path = 'data/DIVERS/coat_ivae/'
    # root_path = 'data/DIVERS/yahoo_ivae/'
    # root_path = 'data/DIVERS/kuai_ivae/'
    # root_path = 'data/DIVERS/sim_ivae/'


    query_embedding_matrix = np.load(root_path+'query_vec1.npy',allow_pickle=True)
    pid_embedding_matrix = np.load(root_path+'embedding_vec1.npy',allow_pickle=True)
    return torch.tensor(query_embedding_matrix).float(), torch.tensor(pid_embedding_matrix).float() 



def load_corresponding_query():

    # corresponding_query = np.load('data/IVERS/coat_ivae/query_vec1.npy', allow_pickle=True)
    # cor_query_pinv = np.load('data/IVERS/coat_ivae/Zt_pinv1.npy', allow_pickle=True)

    # corresponding_query = np.load('data/IVERS/yahoo_ivae/query_vec1.npy', allow_pickle=True)
    # cor_query_pinv = np.load('data/IVERS/yahoo_ivae/Zt_pinv1.npy', allow_pickle=True)

    # corresponding_query = np.load('data/IVERS/kuai_ivae/query_vec1.npy', allow_pickle=True)
    # cor_query_pinv = np.load('data/IVERS/kuai_ivae/Zt_pinv1.npy', allow_pickle=True)

    # corresponding_query = np.load('data/DIVERS/sim_ivae/query_vec1.npy', allow_pickle=True)
    # cor_query_pinv = np.load('data/DIVERS/sim_ivae/Zt_pinv1.npy', allow_pickle=True)



    corresponding_query = np.load('data/DIVERS/coat_ivae/query_vec1.npy', allow_pickle=True)
    # cor_query_pinv = np.load('data/DIVERS/coat_ivae/Zt_pinv1.npy', allow_pickle=True)

    # corresponding_query = np.load('data/DIVERS/yahoo_ivae/query_vec1.npy', allow_pickle=True)
    # cor_query_pinv = np.load('data/DIVERS/yahoo_ivae/Zt_pinv1.npy', allow_pickle=True)

    # corresponding_query = np.load('data/DIVERS/kuai_ivae/query_vec1.npy', allow_pickle=True)
    # cor_query_pinv = np.load('data/DIVERS/kuai_ivae/Zt_pinv1.npy', allow_pickle=True)

    # corresponding_query = np.load('data/DIVERS/sim_ivae/query_vec1.npy', allow_pickle=True)
    # cor_query_pinv = np.load('data/DIVERS/sim_ivae/Zt_pinv1.npy', allow_pickle=True)


    corresponding_query = torch.tensor(corresponding_query).float()


    corresponding_query = corresponding_query.flatten(start_dim=1)


    return corresponding_query



    