

import pandas as pd
import numpy as np
from tqdm import tqdm

input_file = 'user_feat_onehot.csv'
df = pd.read_csv(input_file,header=0)


data_array = df.to_numpy()



data_array1 = data_array
data_array = np.append(data_array, data_array1, axis=0)


print(data_array)

import torch
import torch.nn as nn
import numpy as np

data_array = data_array.astype(np.float32)




data_tensor = torch.tensor(data_array, dtype=torch.float32)



fc_layer = nn.Linear(14, 32)


q_tensor = fc_layer(data_tensor)


q_array = q_tensor.detach().numpy()


print(q_array.shape)

np.save('embedding_vec1.npy', q_array)
np.save('query_vec1.npy', q_array)



query = np.load("query_vec1.npy", allow_pickle=True)
# query_vec.npy is Zt
query = torch.Tensor(query)

Zt = query.unsqueeze(-1)  # np.zeros((query_index.shape[0], 64, top_k))
Zt_pinv = np.zeros((query.shape[0], 1, 32))

for i in tqdm(range(Zt.shape[0])):
    Zt_pinv[i] = np.linalg.pinv(Zt[i])

np.save("./Zt_pinv1.npy", Zt_pinv)