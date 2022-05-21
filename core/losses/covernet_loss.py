from typing import List, Tuple, Callable, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as f

from matplotlib import pyplot as plt


def mean_pointwise_l2_distance(lattice: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Computes the index of the closest trajectory in the lattice as measured by l1 distance.
    :param lattice: Lattice of pre-generated trajectories. Shape [num_modes, n_timesteps, state_dim]
    :param ground_truth: Ground truth trajectory of agent. Shape [1, n_timesteps, state_dim].
    :return: Index of closest mode in the lattice.
    """
    stacked_ground_truth = ground_truth.repeat(lattice.shape[0], 1, 1)
    return torch.pow(lattice - stacked_ground_truth, 2).sum(dim=2).sqrt().mean(dim=1).argmin()


class ConstantLatticeLoss:
    """
    Computes the loss for a constant lattice CoverNet model.
    """

    def __init__(self, lattice: Union[np.ndarray, torch.Tensor],
                 similarity_function: Callable[[torch.Tensor, torch.Tensor], int] = mean_pointwise_l2_distance):
        """
        Inits the loss.
        :param lattice: numpy array of shape [n_modes, n_timesteps, state_dim]
        :param similarity_function: Function that computes the index of the closest trajectory in the lattice
            to the actual ground truth trajectory of the agent.
        """

        self.lattice = torch.Tensor(lattice)
        self.similarity_func = similarity_function

    def visualization(self, trajectories, closest_traj, gt):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        plt.grid()
        ax.set_aspect('equal')

        trajectories = trajectories.cpu().detach().numpy()
        closest_traj = closest_traj.cpu().detach().numpy()

        gt = gt[0].cpu().detach().numpy()

        # for traj in trajectories:
        #     all_x = traj[:, 0]
        #     all_y = traj[:, 1]
        #
        #     plt.plot(all_x, all_y, color='red')
        #     plt.scatter(all_x, all_y, color='red', s=1)

        all_x_closest_traj = closest_traj[:, 0]
        all_y_closest_traj = closest_traj[:, 1]

        plt.plot(all_x_closest_traj, all_y_closest_traj, color='yellow')
        plt.scatter(all_x_closest_traj, all_y_closest_traj, color='yellow', s=1)

        all_x_gt = gt[:, 0]
        all_y_gt = gt[:, 1]

        plt.plot(all_x_gt, all_y_gt, color='green')
        plt.scatter(all_x_gt, all_y_gt, color='green')

        plt.show()

    def get_offset_from_anchors(self, top_trajectories, gt):
        offsets = gt - top_trajectories

        return 0


    def __call__(self, batch_logits: torch.Tensor, batch_ground_truth_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss on a batch.
        :param batch_logits: Tensor of shape [batch_size, n_modes]. Output of a linear layer since this class
            uses nn.functional.cross_entropy.
        :param batch_ground_truth_trajectory: Tensor of shape [batch_size, 1, n_timesteps, state_dim]
        :return: Average element-wise loss on the batch.
        """

        batch_ground_truth_trajectory = torch.cumsum(batch_ground_truth_trajectory, dim=2)

        # If using GPU, need to copy the lattice to the GPU if haven't done so already
        # This ensures we only copy it once
        if self.lattice.device != batch_logits.device:
            self.lattice = self.lattice.to(batch_logits.device)

        batch_losses = torch.Tensor().requires_grad_(True).to(batch_logits.device)

        for logit, ground_truth in zip(batch_logits, batch_ground_truth_trajectory):

            closest_lattice_trajectory = self.similarity_func(self.lattice, ground_truth)

            #self.visualization(self.lattice, self.lattice[closest_lattice_trajectory], ground_truth)

            label = torch.LongTensor([closest_lattice_trajectory]).to(batch_logits.device)
            classification_loss = f.cross_entropy(logit.unsqueeze(0), label)

            self.get_offset_from_anchors((self.lattice[closest_lattice_trajectory]), ground_truth[0])

            batch_losses = torch.cat((batch_losses, classification_loss.unsqueeze(0)), 0)

        return batch_losses.mean()