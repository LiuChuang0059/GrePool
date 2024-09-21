import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from tqdm import tqdm

import torch_geometric.transforms as T


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


class TUUtil:
    @staticmethod
    def add_args(parser):
        parser.set_defaults(batch_size=128)
        parser.set_defaults(epochs=10000)
        parser.set_defaults(lr=0.0005)
        parser.set_defaults(weight_decay=0.0001)
        parser.set_defaults(gnn_dropout=0.5)
        parser.set_defaults(gnn_emb_dim=128)

    @staticmethod
    def loss_fn(task_type):
        def calc_loss(pred, batch, noise=None, m=1.0, uni_rate=1.0):
            loss = F.cross_entropy(pred, batch.y)
            y_uni = torch.ones_like(noise) / (torch.max(batch.y) + 1)
            kl_loss = TUUtil.compute_kl_loss(noise, y_uni)
            return loss + uni_rate * kl_loss

        return calc_loss

    @staticmethod
    def compute_kl_loss(noise, y_uni):
        return F.kl_div(noise.softmax(dim=-1).log(), y_uni.softmax(dim=-1), reduction='batchmean')

    @staticmethod
    @torch.no_grad()
    def eval(model, device, loader, evaluator):
        model.eval()

        correct = 0
        for step, batch in enumerate(tqdm(loader, desc="Eval")):
            batch = batch.to(device)

            pred, noise = model(batch)
            pred = pred.max(dim=1)[1]
            correct += pred.eq(batch.y).sum().item()
        return {"acc": correct / len(loader.dataset)}

    @staticmethod
    def preprocess(args):
        dataset = TUDataset(os.path.join(args.data_root, args.dataset), name=args.dataset)
        if dataset.data.x is None or dataset.data.x.size(1) == 0:
            print("node feature error!")
            max_degree = 0
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if max_degree < 1000:
                dataset.transform = T.OneHotDegree(max_degree)
                dataset = TUDataset(os.path.join(args.data_root, args.dataset), name=args.dataset,
                                    transform=T.OneHotDegree(max_degree))
            else:
                deg = torch.cat(degs, dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                dataset.transform = NormalizedDegree(mean, std)
                dataset = TUDataset(os.path.join(args.data_root, args.dataset), name=args.dataset,
                                    transform=NormalizedDegree(mean, std))
        num_tasks = dataset.num_classes

        num_features = dataset.num_features

        num_training = int(len(dataset) * 0.8)
        num_val = int(len(dataset) * 0.1)
        num_test = len(dataset) - (num_training + num_val)
        training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

        class Dataset(dict):
            pass

        dataset = Dataset({"train": training_set, "valid": validation_set, "test": test_set})
        dataset.eval_metric = "acc"
        dataset.task_type = "classification"
        dataset.get_idx_split = lambda: {"train": "train", "valid": "valid", "test": "test"}

        node_encoder_cls = lambda: nn.Linear(num_features, args.gnn_emb_dim)

        def edge_encoder_cls(_):
            def zero(_):
                return 0

            return zero

        return dataset, num_tasks, node_encoder_cls, edge_encoder_cls, None
