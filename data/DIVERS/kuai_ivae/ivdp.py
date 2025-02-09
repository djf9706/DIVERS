

import pandas as pd
import numpy as np


input_file = 'user_feat_onehot.csv'
df = pd.read_csv(input_file,header=0)


data_array = df.to_numpy()



import torch
import torch.nn as nn
import numpy as np

data_array = data_array.astype(np.float32)


data_tensor = torch.tensor(data_array, dtype=torch.float32)

fc_layer = nn.Linear(121, 5)


q_tensor = fc_layer(data_tensor)


q_array = q_tensor.detach().numpy()

print(q_array.shape)

np.save('embedding_vec1.npy', q_array)
np.save('query_vec1.npy', q_array)
