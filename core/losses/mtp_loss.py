import math
import random
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as f

from matplotlib import pyplot as plt

class MTPLoss:
    """ Computes the loss for the MTP model. """

    def __init__(self,
                 num_modes: int,
                 regression_loss_weight: float = 0.4,
                 angle_threshold_degrees: float = 5.):
        """
        Inits MTP loss.
        :param num_modes: How many modes are being predicted for each agent.
        :param regression_loss_weight: Coefficient applied to the regression loss to
            balance classification and regression performance.
        :param angle_threshold_degrees: Minimum angle needed between a predicted trajectory
            and the ground to consider it a match.
        """
        self.num_modes = num_modes
        self.num_location_coordinates_predicted = 2  # We predict x, y coordinates at each timestep.
        self.regression_loss_weight = regression_loss_weight
        self.angle_threshold = angle_threshold_degrees

    def _get_trajectory_and_modes(self,
                                  model_prediction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the predictions from the model into mode probabilities and trajectory.
        :param model_prediction: Tensor of shape [batch_size, n_timesteps * n_modes * 2 + n_modes].
        :return: Tuple of tensors. First item is the trajectories of shape [batch_size, n_modes, n_timesteps, 2].
            Second item are the mode probabilities of shape [batch_size, num_modes].
        """
        mode_probabilities = model_prediction[:, -self.num_modes:].clone()

        desired_shape = (model_prediction.shape[0], self.num_modes, -1, self.num_location_coordinates_predicted)
        trajectories_no_modes = model_prediction[:, :-self.num_modes].clone().reshape(desired_shape)

        return trajectories_no_modes, mode_probabilities

    @staticmethod
    def _angle_between(ref_traj: torch.Tensor,
                       traj_to_compare: torch.Tensor) -> float:
        """
        Computes the angle between the last points of the two trajectories.
        The resulting angle is in degrees and is an angle in the [0; 180) interval.
        :param ref_traj: Tensor of shape [n_timesteps, 2].
        :param traj_to_compare: Tensor of shape [n_timesteps, 2].
        :return: Angle between the trajectories.
        """

        EPSILON = 1e-5

        if (ref_traj.ndim != 2 or traj_to_compare.ndim != 2 or
                ref_traj.shape[1] != 2 or traj_to_compare.shape[1] != 2):
            raise ValueError('Both tensors should have shapes (-1, 2).')

        if torch.isnan(traj_to_compare[-1]).any() or torch.isnan(ref_traj[-1]).any():
            return 180. - EPSILON

        traj_norms_product = float(torch.norm(ref_traj[-1]) * torch.norm(traj_to_compare[-1]))

        # If either of the vectors described in the docstring has norm 0, return 0 as the angle.
        if math.isclose(traj_norms_product, 0):
            return 0.

        # We apply the max and min operations below to ensure there is no value
        # returned for cos_angle that is greater than 1 or less than -1.
        # This should never be the case, but the check is in place for cases where
        # we might encounter numerical instability.
        dot_product = float(ref_traj[-1].dot(traj_to_compare[-1]))
        angle = math.degrees(math.acos(max(min(dot_product / traj_norms_product, 1), -1)))

        if angle >= 180:
            return angle - EPSILON

        return angle

    @staticmethod
    def _compute_ave_l2_norms(tensor: torch.Tensor) -> float:
        """
        Compute the average of l2 norms of each row in the tensor.
        :param tensor: Shape [1, n_timesteps, 2].
        :return: Average l2 norm. Float.
        """
        l2_norms = torch.norm(tensor, p=2, dim=2)
        avg_distance = torch.mean(l2_norms)
        return avg_distance.item()

    def _compute_angles_from_ground_truth(self, target: torch.Tensor,
                                          trajectories: torch.Tensor) -> List[Tuple[float, int]]:
        """
        Compute angle between the target trajectory (ground truth) and the predicted trajectories.
        :param target: Shape [1, n_timesteps, 2].
        :param trajectories: Shape [n_modes, n_timesteps, 2].
        :return: List of angle, index tuples.
        """
        angles_from_ground_truth = []
        for mode, mode_trajectory in enumerate(trajectories):
            # For each mode, we compute the angle between the last point of the predicted trajectory for that
            # mode and the last point of the ground truth trajectory.
            angle = self._angle_between(target[0], mode_trajectory)

            angles_from_ground_truth.append((angle, mode))
        return angles_from_ground_truth

    def _compute_best_mode(self,
                           angles_from_ground_truth: List[Tuple[float, int]],
                           target: torch.Tensor, trajectories: torch.Tensor) -> int:
        """
        Finds the index of the best mode given the angles from the ground truth.
        :param angles_from_ground_truth: List of (angle, mode index) tuples.
        :param target: Shape [1, n_timesteps, 2]
        :param trajectories: Shape [n_modes, n_timesteps, 2]
        :return: Integer index of best mode.
        """

        # We first sort the modes based on the angle to the ground truth (ascending order), and keep track of
        # the index corresponding to the biggest angle that is still smaller than a threshold value.
        angles_from_ground_truth = sorted(angles_from_ground_truth)
        max_angle_below_thresh_idx = -1
        for angle_idx, (angle, mode) in enumerate(angles_from_ground_truth):
            if angle <= self.angle_threshold:
                max_angle_below_thresh_idx = angle_idx
            else:
                break

        # We choose the best mode at random IF there are no modes with an angle less than the threshold.
        if max_angle_below_thresh_idx == -1:
            best_mode = random.randint(0, self.num_modes - 1)

        # We choose the best mode to be the one that provides the lowest ave of l2 norms between the
        # predicted trajectory and the ground truth, taking into account only the modes with an angle
        # less than the threshold IF there is at least one mode with an angle less than the threshold.
        else:
            # Out of the selected modes above, we choose the final best mode as that which returns the
            # smallest ave of l2 norms between the predicted and ground truth trajectories.
            distances_from_ground_truth = []

            for angle, mode in angles_from_ground_truth[:max_angle_below_thresh_idx + 1]:
                norm = self._compute_ave_l2_norms(target - trajectories[mode, :, :])

                distances_from_ground_truth.append((norm, mode))

            distances_from_ground_truth = sorted(distances_from_ground_truth)
            best_mode = distances_from_ground_truth[0][1]

        return best_mode

    def visualization(self, trajectories, targets):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        trajectories = trajectories[0].detach().numpy()
        targets = targets[0][0].detach().numpy()

        for traj in trajectories:
            all_x = traj[:, 0]
            all_y = traj[:, 1]

            plt.plot(all_x, all_y, color='red')

        all_x_gt = targets[:, 0]
        all_y_gt = targets[:, 1]

        plt.grid()
        plt.xlim([-50, 50])
        plt.ylim([-50, 50])

        plt.plot(all_x_gt, all_y_gt, color='green')

        plt.show()

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the MTP loss on a batch.
        The predictions are of shape [batch_size, n_ouput_neurons of last linear layer]
        and the targets are of shape [batch_size, 1, n_timesteps, 2]
        :param predictions: Model predictions for batch.
        :param targets: Targets for batch.
        :return: zero-dim tensor representing the loss on the batch.
        """

        batch_losses = torch.Tensor().requires_grad_(True).to(predictions.device)
        trajectories, modes = self._get_trajectory_and_modes(predictions)

        trajectories = torch.cumsum(trajectories, dim=2)
        targets = torch.cumsum(targets, dim=2)

        # TODO: Visualization
        #self.visualization(trajectories, targets)

        for batch_idx in range(predictions.shape[0]):

            angles = self._compute_angles_from_ground_truth(target=targets[batch_idx],
                                                            trajectories=trajectories[batch_idx])

            best_mode = self._compute_best_mode(angles,
                                                target=targets[batch_idx],
                                                trajectories=trajectories[batch_idx])

            best_mode_trajectory = trajectories[batch_idx, best_mode, :].unsqueeze(0)

            #regression_loss = f.smooth_l1_loss(best_mode_trajectory, targets[batch_idx])
            regression_loss = f.mse_loss(best_mode_trajectory, targets[batch_idx], reduction='sum')

            mode_probabilities = modes[batch_idx].unsqueeze(0)
            best_mode_target = torch.tensor([best_mode], device=predictions.device)
            classification_loss = f.cross_entropy(mode_probabilities, best_mode_target)

            loss = classification_loss + self.regression_loss_weight * regression_loss

            batch_losses = torch.cat((batch_losses, loss.unsqueeze(0)), 0)

        avg_loss = torch.mean(batch_losses)

        return avg_loss

