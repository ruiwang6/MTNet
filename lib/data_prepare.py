import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def get_dataloaders_from_index_data(
        data_set, steps_per_day, val_ratio, test_ratio, lag, horizon, tod=False, dow=False, dom=False, batch_size=64,
        log=None
):
    data = load_st_dataset(data_set)  # B, N, D

    L, N, F = data.shape
    feature_list = [data]

    # numerical time_in_day
    time_ind = [i % steps_per_day / steps_per_day for i in range(data.shape[0])]
    time_ind = np.array(time_ind)
    time_in_day = np.tile(time_ind, [1, N, 1]).transpose((2, 1, 0))
    feature_list.append(time_in_day)

    # numerical day_in_week
    day_in_week = [(i // steps_per_day) % 7 for i in range(data.shape[0])]
    day_in_week = np.array(day_in_week)
    day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((2, 1, 0))
    feature_list.append(day_in_week)

    data = np.concatenate(feature_list, axis=-1)

    data_train, data_val, data_test = split_data_by_ratio(data, val_ratio, test_ratio)

    # add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, lag, horizon, False)
    y_tra = y_tra[..., :2]
    x_val, y_val = Add_Window_Horizon(data_val, lag, horizon, False)
    y_val = y_val[..., :2]
    x_test, y_test = Add_Window_Horizon(data_test, lag, horizon, False)
    y_test = y_test[..., :2]

    scaler1 = StandardScaler(mean=x_tra[..., 0].mean(), std=x_tra[..., 0].std())
    scaler2 = StandardScaler(mean=x_tra[..., 1].mean(), std=x_tra[..., 1].std())

    x_tra[..., 0] = scaler1.transform(x_tra[..., 0])
    x_val[..., 0] = scaler1.transform(x_val[..., 0])
    x_test[..., 0] = scaler1.transform(x_test[..., 0])

    x_tra[..., 1] = scaler2.transform(x_tra[..., 1])
    x_val[..., 1] = scaler2.transform(x_val[..., 1])
    x_test[..., 1] = scaler2.transform(x_test[..., 1])

    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler1, scaler2

