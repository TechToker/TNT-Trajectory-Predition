import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import copy
import numpy as np
import core.util.visualization as visual

import core.util.drivable_area_polygons_helper as da_helper


class DrivableAreaLoss(nn.Module):

    def __init__(self, query_search_range_manhattan=100, show_visualization=False):
        super(DrivableAreaLoss, self).__init__()

        self.show_visualization = show_visualization
        self.query_search_range_manhattan = query_search_range_manhattan

    # def get_polylines_from_data(self, data):
    #     nodes = data.x.detach().numpy()
    #     cluster = data.cluster.detach().numpy()
    #
    #     polylines = []
    #     for cluster_idc in np.unique(cluster):
    #         [indices] = np.where(cluster == cluster_idc)
    #         polyline = nodes[indices]
    #
    #         polylines.append(polyline)
    #
    #     return polylines

    def forward(self, data, pred_trajs, gt_trajs):

        outside_da_masks = []

        for i in range(data.num_graphs):  # foreach each sample in batch

            pred = pred_trajs[i]
            gt = gt_trajs[i]

            reshape_pred = pred.cpu().detach().numpy().reshape(30, 2).cumsum(axis=0)
            reshape_gt = gt.cpu().numpy().reshape(30, 2).cumsum(axis=0)

            # Works only if batch size = 1 and data not forwarded into VectorNet
            if self.show_visualization:
                data_cpu = data.cpu()
                polylines = da_helper.get_da_polylines_from_data(data_cpu)
                visual.draw_scene(polylines, reshape_gt, reshape_pred)
                #visual.draw_drivable_area(norm_drivable_areas_boundaries, -100, 100, -100, 100)

            drivable_area_polygons = data.clamped_drivable_area[i]
            outline_da_polygon_index = data.outline_drivable_area_index[i]

            outside_da_mask = da_helper.out_of_drivable_area_check(drivable_area_polygons, outline_da_polygon_index, reshape_pred)

            if self.show_visualization:
                visual.draw_scene(polylines, reshape_gt, reshape_pred, outside_da_mask)

            #outside_da_mask = np.repeat(outside_da_mask, 2)  # back to shape 1x60 from 2x30
            outside_da_masks.append(outside_da_mask)

        # get only points which outside drivable area
        total_loss = 0
        for i, traj in enumerate(pred_trajs):
            mask = outside_da_masks[i]
            if not np.any(mask):
                continue

            # Make cumsum
            pred = torch.reshape(pred_trajs[i], (30, 2))
            pred = torch.cumsum(pred, dim=0)

            #pred = torch.reshape(pred, (-1,))

            gt = torch.reshape(gt_trajs[i], (30, 2))
            gt = torch.cumsum(gt, dim=0)
            #gt = torch.reshape(gt, (-1,))

            new_traj = pred[mask]  # pick only points which outside drivable area
            new_gt = gt[mask]

            # if len(new_traj) == 2:
            #     print('here')

            loss = F.mse_loss(new_traj, new_gt, reduction='mean')
            total_loss += loss

        #print(f'out of da loss: {total_loss}')

        return total_loss / len(pred_trajs)



