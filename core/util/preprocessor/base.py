# About:    superclass for data preprocessor
# Author:   Jianbang LIU
# Date:     2021.01.30
import copy
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from argoverse.utils.mpl_plotting_utils import visualize_centerline
import math

class Preprocessor(Dataset):
    """
    superclass for all the trajectory data preprocessor
    those preprocessor will reformat the data in a single sequence and feed to the system or store them
    """

    COUNT_GO_STRAIGHT = 0
    COUNT_LEFT = 0
    COUNT_RIGHT = 0

    COUNT_BEFORE_SAVE_STRAIGHT = 5

    def __init__(self, root_dir, algo="tnt", obs_horizon=20, obs_range=30, pred_horizon=30):
        self.root_dir = root_dir            # root directory stored the dataset

        self.algo = algo                    # the name of the algorithm
        self.obs_horizon = obs_horizon      # the number of timestampe for observation
        self.obs_range = obs_range          # the observation range
        self.pred_horizon = pred_horizon    # the number of timestamp for prediction

        self.split = None

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        """ the total number of sequence in the dataset """
        raise NotImplementedError

    def process(self, dataframe: pd.DataFrame, seq_id: str, map_feat=True):
        """
        select filter the data frame, output filtered data frame
        :param dataframe: DataFrame, the data frame
        :param seq_id: str, the sequence id
        :param map_feat: bool, output map feature or not
        :return: DataFrame[(same as orignal)]
        """
        raise NotImplementedError

    def extract_feature(self, dataframe: pd.DataFrame, map_feat=True):
        """
        select and filter the data frame, output filtered frame feature
        :param dataframe: DataFrame, the data frame
        :param map_feat: bool, output map feature or not
        :return: DataFrame[(same as orignal)]
        """
        raise NotImplementedError

    def encode_feature(self, *feats):
        """
        encode the filtered features to specific format required by the algorithm
        :feats dataframe: DataFrame, the data frame containing the filtered data
        :return: DataFrame[POLYLINE_FEATURES, GT, TRAJ_ID_TO_MASK, LANE_ID_TO_MASK, TARJ_LEN, LANE_LEN]
        """
        raise NotImplementedError

    def save(self, dataframe: pd.DataFrame, file_name, dir_=None):
        """
        save the feature in the data sequence in a single csv files
        :param dataframe: DataFrame, the dataframe encoded
        :param set_name: str, the name of the folder name, exp: train, eval, test
        :param file_name: str, the name of csv file
        :param dir_: str, the directory to store the csv file
        :return:
        """
        if not isinstance(dataframe, pd.DataFrame):
            return

        if not dir_:
            dir_ = os.path.join(os.path.split(self.root_dir)[0], "intermediate", self.split + "_intermediate", "raw")
        else:
            dir_ = os.path.join(dir_, self.split + "_intermediate", "raw")

        if not os.path.exists(dir_):
            os.makedirs(dir_)

        fname = f"features_{file_name}.pkl"
        dataframe.to_pickle(os.path.join(dir_, fname))
        # print("[Preprocessor]: Saving data to {} with name: {}...".format(dir_, fname))


    def balance_data(self, data):

        # HERE CODE TO CALCULATE DIFFERENT BEHAVIOUR
        future_trajectory = data['future_trajectories'][0][0]

        vector_1 = future_trajectory[-1:][0]
        vector_2 = [0, 1]

        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)

        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = math.degrees(np.arccos(dot_product))

        if angle < 5:
            Preprocessor.COUNT_BEFORE_SAVE_STRAIGHT -= 1

            if Preprocessor.COUNT_BEFORE_SAVE_STRAIGHT != 0:
                return False
            else:
                Preprocessor.COUNT_BEFORE_SAVE_STRAIGHT = 5

            Preprocessor.COUNT_GO_STRAIGHT += 1

        if 5 < angle:
            if vector_1[0] < 0:
                Preprocessor.COUNT_LEFT += 1
            else:
                Preprocessor.COUNT_RIGHT += 1

        print(f'STRGT: {Preprocessor.COUNT_GO_STRAIGHT}; LEFT: {Preprocessor.COUNT_LEFT}; RIGHT: {Preprocessor.COUNT_RIGHT}')
        return True

    def process_and_save(self, dataframe: pd.DataFrame, seq_id, dir_=None, map_feat=True):
        """
        save the feature in the data sequence in a single csv files
        :param dataframe: DataFrame, the data frame
        :param set_name: str, the name of the folder name, exp: train, eval, test
        :param file_name: str, the name of csv file
        :param dir_: str, the directory to store the csv file
        :return:
        """
        df_processed = self.process(dataframe, seq_id, map_feat)

        need_save = self.balance_data(df_processed)

        if not need_save:
            return []

        self.save(df_processed, seq_id, dir_)

        return []

    @staticmethod
    def uniform_candidate_sampling(sampling_range, rate=30):
        """
        uniformly sampling of the target candidate
        :param sampling_range: int, the maximum range of the sampling
        :param rate: the sampling rate (num. of samples)
        return rate^2 candidate samples
        """
        x = np.linspace(-sampling_range, sampling_range, rate)
        return np.stack(np.meshgrid(x, x), -1).reshape(-1, 2)

    # implement a candidate sampling with equal distance;
    def lane_points_sampling(self, centerline_list, distance=0.5, viz=False):
        """
        increase amount of centerline points by making interpolation between neighbor points
        :param centerline_list: list of lines, line - array of points
        :param distance: distance step of interpolation algorithm
        return list of all points (w/o deviding into lines)
        """

        candidates = []
        for line in centerline_list:
            for i in range(len(line) - 1):
                if np.any(np.isnan(line[i])) or np.any(np.isnan(line[i+1])):
                    continue

                [x_diff, y_diff] = line[i+1] - line[i]

                if x_diff == 0.0 and y_diff == 0.0:
                    continue

                candidates.append(line[i])

                # compute displacement along each coordinate
                hypot_len = np.hypot(x_diff, y_diff) + np.finfo(float).eps
                d_x = distance * (x_diff / hypot_len)
                d_y = distance * (y_diff / hypot_len)

                num_c = np.floor(hypot_len / distance).astype(np.int)
                pt = copy.deepcopy(line[i])

                # Make interpolation by going from current to next point step-by-step
                for j in range(num_c):
                    pt += np.array([d_x, d_y])
                    candidates.append(copy.deepcopy(pt))

        candidates = np.unique(np.asarray(candidates), axis=0)

        if viz:
            fig = plt.figure(0, figsize=(8, 7))
            fig.clear()
            for centerline_coords in centerline_list:
                visualize_centerline(centerline_coords)
            plt.scatter(candidates[:, 0], candidates[:, 1], marker="*", c="g", alpha=1, s=6.0, zorder=15)
            plt.xlabel("Map X")
            plt.ylabel("Map Y")
            plt.axis("off")
            plt.title("No. of lane candidates = {}; No. of target candidates = {};".format(len(centerline_list), len(candidates)))
            plt.show(block=False)

        return candidates

    @staticmethod
    def get_gt_target_candidate(target_candidates, gt_target):
        """
        find the target candidate closest to the gt and output the one-hot ground truth
        :param target_candidates, (N, 2) candidates
        :param gt_target, (1, 2) the coordinate of final target
        """
        displacement = gt_target - target_candidates

        # find index of the closest target candidate
        closest_target_index = np.argmin(np.power(displacement[:, 0], 2) + np.power(displacement[:, 1], 2))

        onehot = np.zeros((target_candidates.shape[0], 1))
        onehot[closest_target_index] = 1

        offset_xy = gt_target - target_candidates[closest_target_index]
        return onehot, offset_xy

    @staticmethod
    def plot_target_candidates(candidate_centerlines, traj_obs, traj_fut, candidate_targets):
        fig = plt.figure(1, figsize=(8, 7))
        fig.clear()

        # plot traj
        plt.plot(traj_obs[:, 0], traj_obs[:, 1], "x-", color="#d33e4c", alpha=1, linewidth=1, zorder=15)
        # plot end point
        plt.plot(traj_obs[-1, 0], traj_obs[-1, 1], "o", color="#d33e4c", alpha=1, markersize=6, zorder=15)
        # plot future traj
        plt.plot(traj_fut[:, 0], traj_fut[:, 1], "+-", color="b", alpha=1, linewidth=1, zorder=15)

        # plot target sample
        plt.scatter(candidate_targets[:, 0], candidate_targets[:, 1], marker="*", c="green", alpha=1, s=6, zorder=15)

        # plot centerlines
        for centerline_coords in candidate_centerlines:
            visualize_centerline(centerline_coords)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.title("No. of lane candidates = {}; No. of target candidates = {};".format(len(candidate_centerlines),
                                                                                       len(candidate_targets)))
        # plt.show(block=False)
        # plt.pause(0.01)
        plt.show()


# example of preprocessing scripts
if __name__ == "__main__":
    processor = Preprocessor("raw_data")
    loader = DataLoader(processor,
                        batch_size=16,
                        num_workers=16,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False)

    for i, data in enumerate(tqdm(loader)):
        pass



