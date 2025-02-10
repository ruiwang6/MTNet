import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys
import copy

sys.path.append("..")
from lib.utils import (
    MaskedMAELoss,
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data
from model.MTNet import MTLSTformer

@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    val_one_epoch = 0
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        start_time = time.time()
        out_batch = model(x_batch)
        val_one_epoch += time.time() - start_time
        out_batch[..., 0] = SCALER1.inverse_transform(out_batch[..., 0])
        out_batch[..., 1] = SCALER2.inverse_transform(out_batch[..., 1])
        loss1 = criterion(out_batch[..., 0], y_batch[..., 0])
        loss2 = criterion(out_batch[..., 1], y_batch[..., 1])
        loss = loss1 + 2 * loss2
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list), val_one_epoch


@torch.no_grad()
def predict(model, loader):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        out_batch[..., 0] = SCALER1.inverse_transform(out_batch[..., 0])
        out_batch[..., 1] = SCALER2.inverse_transform(out_batch[..., 1])

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out


def train_one_epoch(
        model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=None
):
    global cfg, global_iter_count, global_target_length

    model.train()
    batch_loss_list = []
    one_epoch = 0
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        start_time = time.time()
        out_batch = model(x_batch)
        one_epoch += time.time() - start_time
        out_batch[..., 0] = SCALER1.inverse_transform(out_batch[..., 0])
        out_batch[..., 1] = SCALER2.inverse_transform(out_batch[..., 1])
        loss1 = criterion(out_batch[..., 0], y_batch[..., 0])
        loss2 = criterion(out_batch[..., 1], y_batch[..., 1])
        loss = loss1 + 2 * loss2
        batch_loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss, one_epoch


def train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        clip_grad=0,
        max_epochs=200,
        early_stop=10,
        verbose=1,
        plot=False,
        log=None,
        save=None,
):
    model = model.to(DEVICE)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []
    total_train_time = []
    total_val_time = []
    for epoch in range(max_epochs):
        train_loss, one_epoch = train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=log
        )
        train_loss_list.append(train_loss)
        total_train_time.append(one_epoch)

        val_loss, val_one_epoch = eval_model(model, valset_loader, criterion)
        total_val_time.append(val_one_epoch)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                break
    print(f"Average Training Time: {np.mean(total_train_time)} secs/epoch\n")
    print(f"Average Inference Time: {np.mean(total_val_time)} secs\n")
    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch + 1}\n"
    out_str += f"Best at epoch {best_epoch + 1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if save:
        torch.save(best_state_dict, save)
    return model


@torch.no_grad()
def test_model(model, testset_loader, dataset, log=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, testset_loader)

    end = time.time()

    rmse_all1, mae_all1, mape_all1 = RMSE_MAE_MAPE(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    out_str1 = "Flow All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all1,
        mae_all1,
        mape_all1,
    )
    rmse_all2, mae_all2, mape_all2 = RMSE_MAE_MAPE(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    out_str2 = "Speed All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all2,
        mae_all2,
        mape_all2,
    )

    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :, 0], y_pred[:, i, :, 0])
        out_str1 += "Flow Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :, 1], y_pred[:, i, :, 1])
        out_str2 += "Speed Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str1, log=log, end="")
    print_log(out_str2, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)

    date = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    np.save('./{}_true.npy'.format(dataset + date), y_true)
    np.save('./{}_pred.npy'.format(dataset + date), y_pred)

if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems04")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    args = parser.parse_args()

    seed = torch.randint(1000, (1,))  # set random seed here
    seed_everything(seed)
    set_cpu_num(1)
    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}"
    model_name = MTLSTformer.__name__

    with open(f"{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # -------------------------------- load model -------------------------------- #

    model = MTLSTformer(**cfg["model_args"])

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #

    print_log(dataset, log=log)
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER1, SCALER2
    ) = get_dataloaders_from_index_data(
        dataset,
        steps_per_day=cfg.get("model_args")["steps_per_day"],
        val_ratio=cfg.get("val_size"),
        test_ratio=cfg.get("val_size"),
        lag=cfg.get("in_steps"),
        horizon=cfg.get("out_steps"),
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        log=log,
    )
    print_log(log=log)

    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    if dataset in ("PEMS08", "MANCHESTER","PEMS04"):
        criterion = nn.HuberLoss()
    else:
        raise ValueError("Unsupported dataset.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )

    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log(
        summary(
            model,
            [
                cfg["batch_size"],
                cfg["in_steps"],
                cfg["num_nodes"],
                next(iter(trainset_loader))[0].shape[-1],
            ],
            verbose=0,  # avoid print twice
        ),
        log=log,
    )
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    model = train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        clip_grad=cfg.get("clip_grad"),
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 15),
        verbose=1,
        log=log,
        save=save,
    )

    print_log(f"Saved Model: {save}", log=log)

    test_model(model, testset_loader, dataset, log=log)

    log.close()
