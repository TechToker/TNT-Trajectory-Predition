# VectorNet Implementation
# Author: Jianbang LIU @ RPAI Lab, CUHK
# Email: henryliu@link.cuhk.edu.hk
# Cite: https://github.com/xk-huang/yet-another-vectornet
# Modification: Add auxiliary layer and loss

import os
import copy
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as f
from torch_geometric.data import DataLoader

# from core.model.layers.global_graph import GlobalGraph, SelfAttentionFCLayer
# from core.model.backbone.vectornet import VectorNetBackbone
from core.model.layers.basic_module import MLP
from core.model.backbone.vectornet_v2 import VectorNetBackbone
from core.losses.loss import VectorLoss
from core.losses.mtp_loss import MTPLoss
from core.losses.covernet_loss import ConstantLatticeLoss
from core.losses.drivable_area_loss import DrivableAreaLoss

from core.dataloader.argoverse_loader_v2 import ArgoverseInMem


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

        self.device = device

        self.num_of_modes = 3

        # self.criterion = VectorLoss(with_aux)
        # self.test_criterion = DrivableAreaLoss(show_visualization=False)
        # self.mtp_criterion = MTPLoss(self.num_of_modes)

        PATH_TO_EPSILON_SET = "./trajectory_set/epsilon_8_argo.pkl"
        trajectories = pickle.load(open(PATH_TO_EPSILON_SET, 'rb'))
        self.trajectories_set = torch.Tensor(trajectories)

        self.covernet_criterion = ConstantLatticeLoss(self.trajectories_set)

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
            # nn.Linear(traj_pred_mlp_width, self.horizon * self.out_channels) # For MSE-loss
            # nn.Linear(traj_pred_mlp_width, self.horizon * self.out_channels * self.num_of_modes + self.num_of_modes) # For MTP-loss
            nn.Linear(traj_pred_mlp_width, len(trajectories)) # For CoverNet-loss
        )

    # for MTP
    # def prediction_normalization(self, prediction):
    #     # Normalize the probabilities to sum to 1 for inference
    #     mode_probabilities = prediction[:, -self.num_of_modes:].clone()
    #     if not self.training:
    #         mode_probabilities = f.softmax(mode_probabilities, dim=-1)
    #
    #     predictions = prediction[:, :-self.num_of_modes]
    #
    #     # return pred
    #     return torch.cat((predictions, mode_probabilities), 1)

    def forward(self, data):
        """
        args:
            data (Data): [x, y, cluster, edge_index, valid_len]
        """
        global_feat, _, _ = self.backbone(data) # [batch_size, time_step_len, global_graph_width]
        target_feat = global_feat[:, 0]

        # For MTP
        # pred = self.traj_pred_mlp(target_feat)
        #pred = self.prediction_normalization(pred)
        # end MTP

        logits = self.traj_pred_mlp(target_feat)

        trajectories = []  # shape - [batch_size, top_trajectories]
        probabilities = []  # shape [batch_size, top_probabilities]

        for sample_id in range(len(logits)):
            sorted_logits_indexes = logits[sample_id].argsort(descending=True)

            sorted_logits = logits[sample_id][sorted_logits_indexes]
            sorted_logits = sorted_logits.cpu().detach().numpy()

            sorted_logits_indexes = sorted_logits_indexes.cpu().detach().numpy()
            sorted_trajectories = self.trajectories_set[sorted_logits_indexes]

            trajectories.append(sorted_trajectories[:self.num_of_modes])
            probabilities.append(sorted_logits[:self.num_of_modes])

        return trajectories, probabilities

    def covernet_loss(self, data):

        global_feat, aux_out, aux_gt = self.backbone(data)
        target_feat = global_feat[:, 0]

        pred = self.traj_pred_mlp(target_feat)

        y = data.y.view(-1, self.out_channels * self.horizon)
        y = torch.reshape(y, (len(y), 30, 2))
        y = torch.unsqueeze(y, 1)

        loss = self.covernet_criterion(pred, y)

        return loss

    # def mtp_loss(self, data):
    #     global_feat, aux_out, aux_gt = self.backbone(data)
    #     target_feat = global_feat[:, 0]
    #
    #     pred = self.traj_pred_mlp(target_feat)
    #     pred = self.prediction_normalization(pred)
    #
    #     y = data.y.view(-1, self.out_channels * self.horizon)
    #     y = torch.reshape(y, (len(y), 30, 2))
    #     y = torch.unsqueeze(y, 1)
    #
    #     loss = self.mtp_criterion(pred, y)
    #
    #     return loss

    # def loss(self, data):
    #     global_feat, aux_out, aux_gt = self.backbone(data)
    #     target_feat = global_feat[:, 0]
    #
    #     pred = self.traj_pred_mlp(target_feat)
    #
    #     y = data.y.view(-1, self.out_channels * self.horizon)
    #
    #     return self.criterion(pred, y, aux_out, aux_gt)
    #
    # def loss_override_test(self, data):
    #     copy_data = copy.deepcopy(data)
    #
    #     global_feat, aux_out, aux_gt = self.backbone(data)
    #     target_feat = global_feat[:, 0]
    #
    #     pred = self.traj_pred_mlp(target_feat)
    #     y = data.y.view(-1, self.out_channels * self.horizon)
    #
    #     # da_loss = self.test_criterion(copy_data, pred, y) * 3 # TODO: inside da loss already cumsum => fix it
    #
    #     # Make cum sum for it
    #     pred = torch.reshape(pred, (len(pred), 30, 2))
    #     pred = torch.cumsum(pred, dim=1)
    #     pred = torch.reshape(pred, (len(pred), -1,))
    #
    #     y = torch.reshape(y, (len(y), 30, 2))
    #     y = torch.cumsum(y, dim=1)
    #     y = torch.reshape(y, (len(y), -1,))
    #
    #     mse_loss = self.criterion(pred, y)
    #
    #     #print(f'MSE loss: {mse_loss}; DA loss: {da_loss}')
    #
    #     return mse_loss #+ da_loss

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
