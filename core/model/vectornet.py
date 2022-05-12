# VectorNet Implementation
# Author: Jianbang LIU @ RPAI Lab, CUHK
# Email: henryliu@link.cuhk.edu.hk
# Cite: https://github.com/xk-huang/yet-another-vectornet
# Modification: Add auxiliary layer and loss

import os
import copy
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, DataListLoader, Batch, Data

# from core.model.layers.global_graph import GlobalGraph, SelfAttentionFCLayer
from core.model.layers.global_graph import GlobalGraph, SelfAttentionFCLayer
from core.model.layers.subgraph import SubGraph
from core.dataloader.dataset import GraphDataset, GraphData
# from core.model.backbone.vectornet import VectorNetBackbone
from core.model.layers.basic_module import MLP
from core.model.backbone.vectornet_v2 import VectorNetBackbone
from core.loss import VectorLoss
from core.drivable_area_loss import DrivableAreaLoss
from core.dataloader.argoverse_loader import Argoverse

from core.dataloader.argoverse_loader_v2 import GraphData, ArgoverseInMem


class VectorNet(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self,
                 in_channels=8,
                 horizon=30,
                 num_subgraph_layers=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64,
                 traj_pred_mlp_width=64,
                 with_aux: bool = False,
                 device=torch.device("cpu")):
        super(VectorNet, self).__init__()
        # some params
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)
        self.out_channels = 2
        self.horizon = horizon
        self.subgraph_width = subgraph_width
        self.global_graph_width = global_graph_width
        self.k = 1

        self.device = device

        self.criterion = VectorLoss(with_aux)
        self.test_criterion = DrivableAreaLoss(show_visualization=False)

        # subgraph feature extractor
        self.backbone = VectorNetBackbone(
            in_channels=in_channels,
            num_subgraph_layres=num_subgraph_layers,
            subgraph_width=subgraph_width,
            num_global_graph_layer=num_global_graph_layer,
            global_graph_width=global_graph_width,
            with_aux=with_aux,
            device=device
        )

        # pred mlp
        self.traj_pred_mlp = nn.Sequential(
            MLP(global_graph_width, traj_pred_mlp_width, traj_pred_mlp_width),
            nn.Linear(traj_pred_mlp_width, self.horizon * self.out_channels)
        )

    def forward(self, data):
        """
        args:
            data (Data): [x, y, cluster, edge_index, valid_len]
        """
        global_feat, _, _ = self.backbone(data)              # [batch_size, time_step_len, global_graph_width]
        target_feat = global_feat[:, 0]

        pred = self.traj_pred_mlp(target_feat)

        return pred

    def loss(self, data):
        global_feat, aux_out, aux_gt = self.backbone(data)
        target_feat = global_feat[:, 0]

        pred = self.traj_pred_mlp(target_feat)

        y = data.y.view(-1, self.out_channels * self.horizon)

        return self.criterion(pred, y, aux_out, aux_gt)

    def loss_override_test(self, data):
        copy_data = copy.deepcopy(data)

        global_feat, aux_out, aux_gt = self.backbone(data)
        target_feat = global_feat[:, 0]

        pred = self.traj_pred_mlp(target_feat)
        y = data.y.view(-1, self.out_channels * self.horizon)

        # copy y and make some changes
        # pred = copy.deepcopy(y)
        # for i, point in enumerate(pred[0]):
        #     if i % 2 == 0:
        #         pred[0][i] = pred[0][i] - i * 0.01 # -0.7
        #     # else:
        #     #     pred[i] = 1.22

        #da_loss = self.test_criterion(copy_data, pred, y) * 3 # TODO: inside da loss already cumsum => fix it

        # TODO: Make cum sum for it
        pred = torch.reshape(pred, (len(pred), 30, 2))
        pred = torch.cumsum(pred, dim=1)
        pred = torch.reshape(pred, (len(pred), -1,))

        y = torch.reshape(y, (len(y), 30, 2))
        y = torch.cumsum(y, dim=1)
        y = torch.reshape(y, (len(y), -1,))

        mse_loss = self.criterion(pred, y)

        #print(f'MSE loss: {mse_loss}; DA loss: {da_loss}')

        return mse_loss # + da_loss

    def inference(self, data):
        return self.forward(data)


# %%
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2

    in_channels, pred_len = 10, 30
    show_every = 10
    os.chdir('..')
    # get model
    model = VectorNet(in_channels, pred_len, with_aux=True).to(device)

    DATA_DIR = "../../dataset/interm_data/"
    TRAIN_DIR = os.path.join(DATA_DIR, 'train_intermediate')
    dataset = ArgoverseInMem(TRAIN_DIR)
    data_iter = DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True)

    # train mode
    model.train()
    for i, data in enumerate(data_iter):
        # out, aux_out, mask_feat_gt = model(data)
        loss = model.loss(data.to(device))
        print("Training Pass! loss: {}".format(loss))

        if i == 2:
            break

    # eval mode
    model.eval()
    for i, data in enumerate(data_iter):
        out = model(data.to(device))
        print("Evaluation Pass! Shape of out: {}".format(out.shape))

        if i == 2:
            break
