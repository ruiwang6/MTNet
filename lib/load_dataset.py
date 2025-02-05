import os
import numpy as np

def load_st_dataset(dataset):
    # output B, N, D
    if dataset == "MANCHESTER":
        data_path = os.path.join('../data/Manchester/ManchesterDataFinall.npz')
        data = np.load(data_path)["data"][:, :, 0:2]
        data = np.where(np.isnan(data), 0, data)
    elif dataset == 'PEMS08':
        data_path = os.path.join('../data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, [0, 2]].astype("float32")  # onley the first dimension, traffic flow data
        data = np.where(np.isnan(data), 0, data)
    elif dataset == 'PEMS04':
        data_path = os.path.join('../data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, [0, 2]].astype(
            "float32")  # onley the first dimension, traffic flow data
        data = np.where(np.isnan(data), 0, data)
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    return data