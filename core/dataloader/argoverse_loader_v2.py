import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import gc
from copy import deepcopy, copy

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
# from torch.utils.data import DataLoader

sys.path.append("core/dataloader")


def get_fc_edge_index(node_indices):
    """
    node_indices: np.array([indices]), the indices of nodes connecting with each other;
    return a tensor(2, edges), indicing edge_index

    i.e return 2dim array where rows represent connection between each index (from node_indices) (order matter)
    format : 0 - 0, 1 - 0, 2 - 0, 1 - 0, 1 - 1, 1 - 2, 2 - 0, 2 - 1, 2 - 2
    """
    xx, yy = np.meshgrid(node_indices, node_indices)
    xy = np.vstack(([xx.reshape(-1), yy.reshape(-1)])).astype(np.int64)
    return xy


def get_traj_edge_index(node_indices):
    """
    generate the polyline graph for traj, each node are only directionally connected with the nodes in its future
    node_indices: np.array([indices]), the indices of nodes connecting with each other;
    return a tensor(2, edges), indicing edge_index;

    i.e return 2dim array where rows represent connection between each index (from node_indices) (order dont matter)
    format : 0 - 0, 0 - 1, 0 - 2, 1 - 1, 1 - 2, 2 - 2
    """
    edge_index = np.empty((2, 0))
    for i in range(len(node_indices)):
        xx, yy = np.meshgrid(node_indices[i], node_indices[i:])
        edge_index = np.hstack([edge_index, np.vstack(([xx.reshape(-1), yy.reshape(-1)])).astype(np.int64)])
    return edge_index


class GraphData(Data):
    """
    override graphs mini-batching logic: override key `cluster` indicating which polyline_id is for the vector
    """

    def __inc__(self, key, value):
        if key == 'edge_index':
            return self.x.size(0)
        elif key == 'cluster':
            return int(self.cluster.max().item()) + 1
        else:
            return 0

# %%


# dataset loader which loads data into memory
class ArgoverseInMem(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ArgoverseInMem, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        gc.collect()

    @property
    def raw_file_names(self):
        return [file for file in os.listdir(self.raw_dir) if "features" in file and file.endswith(".pkl")]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        """ transform the raw data and store in GraphData """
        # loading the raw data
        traj_lens = []
        valid_lens = []
        candidate_lens = []

        for raw_path in tqdm(self.raw_paths, desc="Loading Raw Data..."):
            raw_data = pd.read_pickle(raw_path)

            # statistics
            traj_num = raw_data['all_agents_history'].values[0].shape[0] # amount of trajectories in dataset
            traj_lens.append(traj_num)

            lane_num = raw_data['graph'].values[0]['lane_idcs'].max() + 1 # amount of lanes in dataset
            valid_lens.append(traj_num + lane_num)

            candidate_num = raw_data['target_candidates'].values[0].shape[0]
            candidate_lens.append(candidate_num)

        num_valid_len_max = np.max(valid_lens)
        num_candidate_max = np.max(candidate_lens)
        print("\n[Argoverse]: The maximum of valid length is {}.".format(num_valid_len_max))
        print("[Argoverse]: The maximum of no. of candidates is {}.".format(num_candidate_max))

        # pad vectors to the largest polyline id and extend cluster, save the Data to disk
        data_list = []
        for ind, raw_path in enumerate(tqdm(self.raw_paths, desc="Transforming the data to GraphData...")):
            raw_data = pd.read_pickle(raw_path)

            # input data
            x, cluster, edge_index, identifier = self._get_x(raw_data)
            y = self._get_y(raw_data)

            # torch geometric graph data
            graph_input = GraphData(
                x=torch.from_numpy(x).float(), # node feature matrix. All lines/trajectories with 10 features [num_nodes, num_node_features]
                y=torch.from_numpy(y).float(), # get offsets from point to point over the future trajectory. shape=60 (30 time x 2 cord)
                cluster=torch.from_numpy(cluster).short(), # array which define belongings each node to polyline
                edge_index=torch.from_numpy(edge_index).long(), # graph connectivity in coordinate format [2, num_edges]. Connectivity between nodes by indexes
                identifier=torch.from_numpy(identifier).float(),    # identify embedding of global graph completion

                traj_len=torch.tensor([traj_lens[ind]]).int(),         # number of traj polyline on each scene
                valid_len=torch.tensor([valid_lens[ind]]).int(),       # number of traj + lanes polylines on each scene
                time_step_len=torch.tensor([num_valid_len_max]).int(), # the maximum of number of polyline over whole dataset

                candidate_len_max=torch.tensor([num_candidate_max]).int(), # max amount of target candidates over whole dataset
                candidate_mask=[],
                candidate=torch.from_numpy(raw_data['target_candidates'].values[0]).float(), # target
                candidate_gt=torch.from_numpy(raw_data['target_candidates_onehot'].values[0]).bool(),
                offset_gt=torch.from_numpy(raw_data['target_offset_gt'].values[0]).float(),
                target_gt=torch.from_numpy(raw_data['future_trajectories'].values[0][0][-1, :]).float(),

                orig=torch.from_numpy(raw_data['origin_pos'].values[0]).float().unsqueeze(0), # prediction start position
                rot=torch.from_numpy(raw_data['rotation_matrix'].values[0]).float().unsqueeze(0), # rotation matrices for current scene
                seq_id=torch.tensor([int(raw_data['seq_id'])]).int()
            )

            data_list.append(graph_input)

        # Because saving a huge python list is rather slow, we collate the list into one huge Data
        # slices dictionary to reconstruct single examples from this object
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = super(ArgoverseInMem, self).get(idx).clone()

        feature_len = data.x.shape[1] # get amount of features per node
        index_to_pad = data.time_step_len[0].item() # get maximum of amount of polyline on scene
        valid_len = data.valid_len[0].item() # get number of polylines on scene

        # pad feature with zero nodes (fill to full-fize tensor)
        data.x = torch.cat([data.x, torch.zeros((index_to_pad - valid_len, feature_len), dtype=data.x.dtype)])
        data.cluster = torch.cat([data.cluster, torch.arange(valid_len, index_to_pad)]) # full fill cluster (subgraph)
        data.identifier = torch.cat([data.identifier, torch.zeros((index_to_pad - valid_len, 2), dtype=data.x.dtype)])

        # pad candidate and candidate_gt
        num_cand_max = data.candidate_len_max[0].item()
        data.candidate_mask = torch.cat([torch.ones((len(data.candidate), 1)),
                                         torch.zeros((num_cand_max - len(data.candidate), 1))])
        data.candidate = torch.cat([data.candidate, torch.zeros((num_cand_max - len(data.candidate), 2))])
        data.candidate_gt = torch.cat([data.candidate_gt, torch.zeros((num_cand_max - len(data.candidate_gt), 1))])

        return data

    @staticmethod
    def _get_x(data_seq):
        """
        feat: [xs, ys, vec_x, vec_y, step(timestamp), traffic_control, turn, is_intersection, polyline_id];
        xs, ys: the control point of the vector, for trajectory, it's start point, for lane segment, it's the center point;
        vec_x, vec_y: the length of the vector in x, y coordinates;
        step: indicating the step of the trajectory, for the lane node, it's always 0;
        traffic_control: feature for lanes. for trajectories always 0;
        turn: two binary indicator representing is the lane turning left or right; for trajectories always 0;
        is_intersection: indicating whether the lane segment is in intersection; for trajectories always 0;
        polyline_id: the polyline id of this node belonging to;

        identifier: # vstask of min_x and min_y position for every trajectory or line # TODO: Why separately
        """
        feats = np.empty((0, 10))
        edge_index = np.empty((2, 0), dtype=np.int64)
        identifier = np.empty((0, 2))

        # get traj features
        traj_feats = data_seq['all_agents_history'].values[0]
        traj_has_obss = data_seq['agents_history_presence'].values[0]

        step = np.arange(0, traj_feats.shape[1]).reshape((-1, 1))
        traj_cnt = 0

        # foreach for every agent trajectory
        # vstask every trajectory
        for _, [feat, has_obs] in enumerate(zip(traj_feats, traj_has_obss)):
            xy_s = feat[has_obs][:-1, :2] # remove last element from trajectory history, get only x, y cords
            vec = feat[has_obs][1:, :2] - feat[has_obs][:-1, :2] # length from point to point of trajectory

            # for trajectory traffic/intersect/turn equals 0
            traffic_ctrl = np.zeros((len(xy_s), 1))
            is_intersect = np.zeros((len(xy_s), 1))
            is_turn = np.zeros((len(xy_s), 2))

            polyline_id = np.ones((len(xy_s), 1)) * traj_cnt
            feats = np.vstack([feats, np.hstack([xy_s, vec, step[has_obs][:-1], traffic_ctrl, is_turn, is_intersect, polyline_id])])
            traj_cnt += 1

        # get lane features
        graph = data_seq['graph'].values[0]
        ctrs = graph['centers']
        vec = graph['lines_vectors']
        traffic_ctrl = graph['lines_traffic_control_info'].reshape(-1, 1)
        is_turns = graph['lines_turn_info']
        is_intersect = graph['lines_intersect_info'].reshape(-1, 1)
        lane_idcs = graph['lane_idcs'].reshape(-1, 1) + traj_cnt

        steps = np.zeros((len(lane_idcs), 1))

        # vstack lines info to trajectory info
        feats = np.vstack([feats, np.hstack([ctrs, vec, steps, traffic_ctrl, is_turns, is_intersect, lane_idcs])])

        # get the cluster and construct subgraph edge_index
        cluster = copy(feats[:, -1].astype(np.int64)) # copy of last column of feats (with id)
        for cluster_idc in np.unique(cluster):
            [indices] = np.where(cluster == cluster_idc)

            # vstask min_x and min_y position for every trajectory or line #TODO: Why separately
            identifier = np.vstack([identifier, np.min(feats[indices, :2], axis=0)])

            if len(indices) <= 1:
                continue                # skip if only 1 node

            if cluster_idc < traj_cnt:
                # if it's a trajectory
                edge_index = np.hstack([edge_index, get_traj_edge_index(indices)])
            else:
                # if it's a line
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)])

        return feats, cluster, edge_index, identifier

    @staticmethod
    def _get_y(data_seq):
        traj_obs = data_seq['all_agents_history'].values[0][0]
        traj_fut = data_seq['future_trajectories'].values[0][0]
        offset_fut = np.vstack([traj_fut[0, :] - traj_obs[-1, :2], traj_fut[1:, :] - traj_fut[:-1, :]])
        return offset_fut.reshape(-1).astype(np.float32)


if __name__ == "__main__":

    # for folder in os.listdir("./data/interm_data"):
    INTERMEDIATE_DATA_DIR = "../../dataset/interm_data_small"
    # INTERMEDIATE_DATA_DIR = "../../dataset/interm_tnt_n_s_0804"
    # INTERMEDIATE_DATA_DIR = "/media/Data/autonomous_driving/Argoverse/intermediate"

    for folder in ["train", "val", "test"]:
    # for folder in ["test"]:
        dataset_input_path = os.path.join(INTERMEDIATE_DATA_DIR, f"{folder}_intermediate")

        # dataset = Argoverse(dataset_input_path)
        dataset = ArgoverseInMem(dataset_input_path) #.shuffle()
        batch_iter = DataLoader(dataset, batch_size=16, num_workers=16, shuffle=False, pin_memory=True)
        for k in range(1):
            for i, data in enumerate(tqdm(batch_iter, total=len(batch_iter), bar_format="{l_bar}{r_bar}")):
                pass

            # print("{}".format(i))
            # candit_len = data.candidate_len_max[0]
            # print(candit_len)
            # target_candite = data.candidate[candit_gt.squeeze(0).bool()]
            # try:
            #     # loss = torch.nn.functional.binary_cross_entropy(candit_gt, candit_gt)
            #     target_candite = data.candidate[candit_gt.bool()]
            # except:
            #     print(torch.argmax())
            #     print(candit_gt)
            # # print("type: {}".format(type(candit_gt)))
            # print("max: {}".format(candit_gt.max()))
            # print("min: {}".format(candit_gt.min()))

