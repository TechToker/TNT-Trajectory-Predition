# About: script to processing argoverse forecasting dataset
# Author: Jianbang LIU @ RPAI, CUHK
# Date: 2021.07.16

import os
import argparse
from os.path import join as pjoin
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import sparse

import warnings

# import torch
from torch.utils.data import Dataset, DataLoader

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.utils.mpl_plotting_utils import visualize_centerline

from core.util.preprocessor.base import Preprocessor
from core.util.cubic_spline import Spline2D

warnings.filterwarnings("ignore")


class ArgoversePreprocessor(Preprocessor):
    def __init__(self,
                 root_dir,
                 split="train",
                 algo="tnt",
                 obs_horizon=20,
                 obs_range=100,
                 pred_horizon=30,
                 normalized=True,
                 save_dir=None):
        super(ArgoversePreprocessor, self).__init__(root_dir, algo, obs_horizon, obs_range, pred_horizon)

        self.LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
        self.COLOR_DICT = {"AGENT": "#d33e4c", "OTHERS": "#489ab5", "AV": "#007672"}

        self.split = split
        self.normalized = normalized

        self.am = ArgoverseMap()
        self.loader = ArgoverseForecastingLoader(pjoin(self.root_dir, self.split+"_obs" if split == "test" else split))

        self.save_dir = save_dir

    # Method only save dataframe sequence into
    def __getitem__(self, idx):
        #print(f'get item: {idx}')
        f_path = self.loader.seq_list[idx]

        seq = self.loader.get(f_path)
        path, seq_f_name_ext = os.path.split(f_path)
        seq_f_name, ext = os.path.splitext(seq_f_name_ext)

        #print(f'path: {seq_f_name}')

        df = copy.deepcopy(seq.seq_df)

        return self.process_and_save(df, seq_id=seq_f_name, dir_=self.save_dir)

    # Process original Argoverse dataframe
    def process(self, dataframe: pd.DataFrame, seq_id, map_feat=True):

        data = self.read_argo_data(dataframe)
        data = self.get_obj_feats(data)

        data['graph'] = self.get_lane_graph(data)
        data['seq_id'] = seq_id

        # visualization for debug purpose
        #self.visualize_data(data)

        # Convert from dictionary to dataframe
        result = pd.DataFrame([[data[key] for key in data.keys()]], columns=[key for key in data.keys()])
        return result

    def __len__(self):
        return len(self.loader)

    """
        Convert dataframe to format <city_name, list with trajectory for all agents, timestamps indexes for each position>
    """
    @staticmethod
    def read_argo_data(df: pd.DataFrame):
        # df format: TIMESTAMP, TRACK_ID, OBJECT_TYPE, X, Y, CITY_NAME
        city = df["CITY_NAME"].values[0]

        all_timestamps = np.sort(np.unique(df['TIMESTAMP'].values))

        mapping = dict()
        for i, timestamp in enumerate(all_timestamps):
            mapping[timestamp] = i

        all_agents_trajectories = np.concatenate((df.X.to_numpy().reshape(-1, 1), df.Y.to_numpy().reshape(-1, 1)), 1)

        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        agt_idx = obj_type.index('AGENT')
        idcs = objs[keys[agt_idx]]

        agt_traj = all_agents_trajectories[idcs]
        agt_step = steps[idcs]

        del keys[agt_idx]
        ctx_trajs, ctx_steps = [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(all_agents_trajectories[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data['city'] = city
        data['all_agents_trajectories'] = [agt_traj] + ctx_trajs
        data['agents_timestamp_presence'] = [agt_step] + ctx_steps

        return data

    def get_rotation_matrix(self, data, trgt_agent_current_position):
        if self.normalized:
            pre, conf = self.am.get_lane_direction(data['all_agents_trajectories'][0][self.obs_horizon - 1], data['city'])

            if conf <= 0.1:
                pre = (trgt_agent_current_position - data['all_agents_trajectories'][0][self.obs_horizon - 4]) / 2.0

            theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2

            rot = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], np.float32)
        else:
            # if not normalized, do not rotate.
            theta = None
            rot = np.asarray([
                [1.0, 0.0],
                [0.0, 1.0]], np.float32)

        return theta, rot

    @staticmethod
    def get_ref_centerline(centerlines, pred_gt):
        """
            return: centerlines as lines of Spline2D, and index of closest centerline
        """
        if len(centerlines) == 1:
            return [Spline2D(x=centerlines[0][:, 0], y=centerlines[0][:, 1])], 0
        else:
            ref_centerlines = [Spline2D(x=centerlines[i][:, 0], y=centerlines[i][:, 1]) for i in range(len(centerlines))]

            # search the closest point of the traj final position to each center line
            min_distances = []
            for line in ref_centerlines:
                xy = np.stack([line.x_fine, line.y_fine], axis=1)
                diff = xy - pred_gt[-1, :2]
                dis = np.hypot(diff[:, 0], diff[:, 1])
                min_distances.append(np.min(dis))

            line_idx = np.argmin(min_distances)
            return ref_centerlines, line_idx

    def get_target_agent_features(self, city_name, target_agent_full_trajectory, rot, origin_position):
        # get the target candidates and candidate gt
        agent_history_trajectory = target_agent_full_trajectory[0: self.obs_horizon]
        agent_future_trajectory = target_agent_full_trajectory[self.obs_horizon:self.obs_horizon+self.pred_horizon].copy().astype(np.float32)

        centerline_candidates = self.am.get_candidate_centerlines_for_traj(agent_history_trajectory, city_name)

        # rotate the center lines and find the reference center line
        agent_history_trajectory = np.matmul(rot, (agent_history_trajectory - origin_position.reshape(-1, 2)).T).T
        agent_future_trajectory = np.matmul(rot, (agent_future_trajectory - origin_position.reshape(-1, 2)).T).T

        for i, _ in enumerate(centerline_candidates):
            centerline_candidates[i] = np.matmul(rot, (centerline_candidates[i] - origin_position.reshape(-1, 2)).T).T

        target_candidates = self.lane_points_sampling(centerline_candidates) #, viz=True)

        # Because test labels not available
        if self.split == "test":
            target_candidates_onehot, target_offset_gt = np.zeros((target_candidates.shape[0], 1)), np.zeros((1, 2))
            centerline_splines, reference_centerline_idx = None, None
        else:
            target_candidates_onehot, target_offset_gt = self.get_gt_target_candidate(target_candidates, agent_future_trajectory[-1])
            centerline_splines, reference_centerline_idx = self.get_ref_centerline(centerline_candidates, agent_future_trajectory)

        # Visualize target candidates
        #self.plot_target_candidates(centerline_candidates, agent_history_trajectory, agent_future_trajectory, target_candidates)

        return target_candidates, target_candidates_onehot, target_offset_gt, centerline_splines, reference_centerline_idx

    def get_obj_feats(self, data):
        # get the origin (start pos) and compute the orientation of the target agent
        origin_position = data['all_agents_trajectories'][0][self.obs_horizon-1].copy().astype(np.float32)

        # compute the rotation matrix
        rotation_angle, rotation_matrix = self.get_rotation_matrix(data, origin_position)

        history_trajectories, agents_history_presence, future_trajectories, agents_future_presence = [], [], [], []
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range

        for traj, step in zip(data['all_agents_trajectories'], data['agents_timestamp_presence']):
            # if agent not present on current step
            if self.obs_horizon-1 not in step:
                continue

            # Trajectory normalize and rotate
            normalized_trajectory = np.matmul(rotation_matrix, (traj - origin_position.reshape(-1, 2)).T).T

            # collect the future trajectory (ground truth)
            gt_trajectory = np.zeros((self.pred_horizon, 2), np.float32)

            # Init masks. # We use masks because at some timestamp agent may not be present
            is_has_gt_on_timestamp = np.zeros(self.pred_horizon, np.bool)
            is_timestamp_future = np.logical_and(step >= self.obs_horizon, step < self.obs_horizon + self.pred_horizon)

            future_steps = step[is_timestamp_future] - self.obs_horizon
            post_trajectory = normalized_trajectory[is_timestamp_future]
            gt_trajectory[future_steps] = post_trajectory
            is_has_gt_on_timestamp[future_steps] = True

            # collect the history
            is_timestamp_history = step < self.obs_horizon
            history_steps = step[is_timestamp_history]
            history_trajectory = normalized_trajectory[is_timestamp_history]

            # TODO: Why argsort. It must be sorted already
            idcs = history_steps.argsort()
            history_steps = history_steps[idcs]
            history_trajectory = history_trajectory[idcs]

            for i in range(len(history_steps)):
                if history_steps[i] == self.obs_horizon - len(history_steps) + i:
                    break

            # TODO: Test is any cases in which after cycle i will be '!=0'
            # if i != 0:
            #     print('BIG ERROR')

            history_steps = history_steps[i:]
            history_trajectory = history_trajectory[i:]

            if len(history_steps) <= 1:
                #print(f'PROBLEM: history len = {len(history_steps)}')
                continue

            history_trajectory_3D = np.zeros((self.obs_horizon, 3), np.float32)
            history_trajectory_3D[history_steps, 2] = 1.0
            history_trajectory_3D[history_steps, :2] = history_trajectory

            is_has_history_on_timestamp = np.zeros(self.obs_horizon, np.bool)
            is_has_history_on_timestamp[history_steps] = True

            if history_trajectory_3D[-1, 0] < x_min or history_trajectory_3D[-1, 0] > x_max or history_trajectory_3D[-1, 1] < y_min or history_trajectory_3D[-1, 1] > y_max:
                #print(f'PROBLEM: last history coordinate out of the bounds')
                continue

            history_trajectories.append(history_trajectory_3D)
            future_trajectories.append(gt_trajectory)

            agents_history_presence.append(is_has_history_on_timestamp)
            agents_future_presence.append(is_has_gt_on_timestamp)

        history_trajectories = np.asarray(history_trajectories, np.float32)
        agents_history_presence = np.asarray(agents_history_presence, np.bool)
        future_trajectories = np.asarray(future_trajectories, np.float32)
        agents_future_presence = np.asarray(agents_future_presence, np.bool)

        # plot the splines
        # self.plot_reference_centerlines(ctr_line_candts, splines, feats[0], gt_preds[0], ref_idx)

        data['origin_pos'] = origin_position
        data['rotation_angle'] = rotation_angle
        data['rotation_matrix'] = rotation_matrix

        data['all_agents_history'] = history_trajectories
        data['agents_history_presence'] = agents_history_presence

        data['agents_future_presence'] = agents_future_presence
        data['future_trajectories'] = future_trajectories

        target_agent_full_trajectory = data['all_agents_trajectories'][0].copy().astype(np.float32)
        target_candidates, target_candidates_onehot, target_offset_gt, centerline_splines, reference_centerline_idx \
            = self.get_target_agent_features(data['city'],
                                             target_agent_full_trajectory,
                                             rotation_matrix,
                                             origin_position)

        data['target_candidates'] = target_candidates
        data['target_candidates_onehot'] = target_candidates_onehot
        data['target_offset_gt'] = target_offset_gt

        data['centerline_splines'] = centerline_splines  # the reference candidate centerlines Spline for prediction
        data['reference_centerline_idx'] = reference_centerline_idx  # the idx of the closest reference centerlines

        return data

    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))

        lane_ids_in_roi = self.am.get_lane_ids_in_xy_bbox(data['origin_pos'][0], data['origin_pos'][1], data['city'], radius * 1.5)
        lane_ids_in_roi = copy.deepcopy(lane_ids_in_roi)

        lanes = dict()  # dictionary of <lane_id, lane segment>
        for lane_id in lane_ids_in_roi:
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)

            centerline = np.matmul(data['rotation_matrix'], (lane.centerline - data['origin_pos'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]

            # if some point of the line outside roi
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)

                lane.centerline = centerline
                lane.polygon = np.matmul(data['rotation_matrix'], (polygon[:, :2] - data['origin_pos'].reshape(-1, 2)).T).T

                lanes[lane_id] = lane

        lane_ids = list(lanes.keys())
        lines_vectors_centers, lines_vectors, lines_turn_info, lines_traffic_control_info, lines_intersect_info = [], [], [], [], []

        for lane_id in lane_ids:
            lane = lanes[lane_id]
            centerline = lane.centerline
            num_segs = len(centerline) - 1

            # array of center points (between points of centerline)
            centers = np.asarray((centerline[:-1] + centerline[1:]) / 2.0, np.float32)
            lines_vectors_centers.append(centers)

            # Vectors that make up the line (displacements)
            lin_vectors = np.asarray(centerline[1:] - centerline[:-1], np.float32)
            lines_vectors.append(lin_vectors)

            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass

            lines_turn_info.append(x)

            lines_traffic_control_info.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            lines_intersect_info.append(lane.is_intersection * np.ones(num_segs, np.float32))

        lane_idcs = []
        count = 0
        for i, ctr in enumerate(lines_vectors_centers):
            lane_idcs.append(i * np.ones(len(ctr), np.int64))
            count += len(ctr)
        num_nodes = count

        graph = dict()
        graph['lane_idcs'] = np.concatenate(lane_idcs, 0)
        graph['num_nodes'] = num_nodes

        graph['centers'] = np.concatenate(lines_vectors_centers, 0)
        graph['lines_vectors'] = np.concatenate(lines_vectors, 0)
        graph['lines_turn_info'] = np.concatenate(lines_turn_info, 0)
        graph['lines_traffic_control_info'] = np.concatenate(lines_traffic_control_info, 0)
        graph['lines_intersect_info'] = np.concatenate(lines_intersect_info, 0)

        return graph

    def plot_reference_centerlines(self, cline_list, splines, obs, pred, ref_line_idx):
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        for centerline_coords in cline_list:
            visualize_centerline(centerline_coords)

        for i, spline in enumerate(splines):
            xy = np.stack([spline.x_fine, spline.y_fine], axis=1)
            if i == ref_line_idx:
                plt.plot(xy[:, 0], xy[:, 1], "--", color="r", alpha=0.7, linewidth=1, zorder=10)
            else:
                plt.plot(xy[:, 0], xy[:, 1], "--", color="b", alpha=0.5, linewidth=1, zorder=10)

        self.plot_traj(obs, pred)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.show()
        # plt.show(block=False)
        # plt.pause(0.5)

    def plot_traj(self, obs, pred, traj_id=None):
        assert len(obs) != 0, "ERROR: The input trajectory is empty!"
        traj_na = "t{}".format(traj_id) if traj_id else "traj"
        obj_type = "AGENT" if traj_id == 0 else "OTHERS"

        plt.plot(obs[:, 0], obs[:, 1], color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)
        plt.plot(pred[:, 0], pred[:, 1], "d-", color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)

        plt.text(obs[0, 0], obs[0, 1], "{}_s".format(traj_na))

        if len(pred) == 0:
            plt.text(obs[-1, 0], obs[-1, 1], "{}_e".format(traj_na))
        else:
            plt.text(pred[-1, 0], pred[-1, 1], "{}_e".format(traj_na))

    def visualize_data(self, data):
        """
        visualize the extracted data, and exam the data
        """
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        # visualize the centerlines
        lines_vectors_centers = data['graph']['centers']
        lines_vectors = data['graph']['lines_vectors']
        lane_idcs = data['graph']['lane_idcs']

        for i in np.unique(lane_idcs):
            line_vectors_centers = lines_vectors_centers[lane_idcs == i]
            line_vectors = lines_vectors[lane_idcs == i]

            line_str = (2.0 * line_vectors_centers - line_vectors) / 2.0
            line_end = (2.0 * line_vectors_centers[-1, :] + line_vectors[-1, :]) / 2.0

            line = np.vstack([line_str, line_end.reshape(-1, 2)])
            visualize_centerline(line)

        # visualize the trajectory
        trajs = data['all_agents_history'][:, :, :2]
        has_obss = data['agents_history_presence']
        preds = data['future_trajectories']
        has_preds = data['agents_future_presence']

        for i, [traj, has_obs, pred, has_pred] in reversed(list(enumerate(zip(trajs, has_obss, preds, has_preds)))):
            self.plot_traj(traj[has_obs], pred[has_pred], i)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.show()


def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


# Generate a small subset to test the training program
# Get all data from agroverse dataset
# Get trajectories and centerlines
# Transform trajectories/centerlines to local (target agent) frame
# Save it as .pkl file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-r", "--root", type=str, default="../dataset")
    # parser.add_argument("-d", "--dest", type=str, default="../dataset")
    # parser.add_argument("-s", "--small", action='store_true', default=False)

    parser.add_argument("-r", "--root", type=str, default="/home/techtoker/projects/TNT-Trajectory-Predition/dataset/")
    parser.add_argument("-d", "--dest", type=str, default="dataset")
    parser.add_argument("-s", "--small", action='store_true', default=True)

    args = parser.parse_args()

    print('Args:')
    print(args)
    print()

    raw_dir = os.path.join(args.root, "raw_data")
    interm_dir = os.path.join(args.dest, "interm_data" if not args.small else "interm_data_small")

    print(raw_dir, interm_dir)
    print()

    # Foreach train / val / test split
    for split in ["train", "val", "test"]:
        # construct the preprocessor and dataloader
        print(raw_dir)
        print()
        argoverse_processor = ArgoversePreprocessor(root_dir=raw_dir, split=split, save_dir=interm_dir)

        #item = argoverse_processor[0]

        loader = DataLoader(argoverse_processor,
                            batch_size=1,
                            num_workers=1,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)

        for i, data in enumerate(tqdm(loader)):

            # if split == "test":
            #     break

            if args.small:
                if split == "train" and i >= 250:
                    break
                elif split == "val" and i >= 100:
                    break
                elif split == "test" and i >= 100:
                    break
