import os
import csv
import random
import torch
import numpy as np
from pynvml import *


def num_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_gpu_utilization(device_index):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_index)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def seed_everything(seed):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
        Python.

        Args:
            seed (int): The desired seed.
        """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def results_to_file(args, test_acc, test_std,
                    val_acc, val_std,
                    total_time, total_time_std,
                    avg_time, avg_time_std):
    if not os.path.exists('./results/{}'.format(args.dataset)):
        print("=" * 20)
        print("Creating Results File !!!")

        os.makedirs('./results/{}'.format(args.dataset))

    filename = "./results/{}/result_seed{}.csv".format(
        args.dataset, args.seed)

    headerList = ["Method", "Token_Remain_Ratio",
                  "Encoder_Layers", "Uniform_Ratio",
                  "Attn_Dropout_Ratio",
                  "::::::::",
                  "test_acc", "test_std",
                  "val_acc", "val_std",
                  "total_time", "total_time_std",
                  "avg_time", "avg_time_std"]

    with open(filename, "a+") as f:

        # reader = csv.reader(f)
        # row1 = next(reader)
        f.seek(0)
        header = f.read(6)
        if header != "Method":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()

        line = "{}, {}, {}, {}, {}, :::::::::, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(
            args.model_type, args.token_ratio,
            args.num_encoder_layers, args.uniform_rate,
            args.dropout_attn,
            test_acc, test_std,
            val_acc, val_std,
            total_time, total_time_std,
            avg_time, avg_time_std
        )
        f.write(line)
