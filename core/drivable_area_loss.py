import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import copy
import numpy as np
import core.util.visualization as visual
from argoverse.map_representation.map_api import ArgoverseMap

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from matplotlib.patches import Polygon as plotpoly

from matplotlib import pyplot as plt

class DrivableAreaLoss(nn.Module):

    def __init__(self):
        super(DrivableAreaLoss, self).__init__()
        self.am = ArgoverseMap()

        self.drivable_area_outline_pit = self.am.get_vector_map_driveable_areas('PIT')[0]
        self.drivable_area_outline_mia = self.am.get_vector_map_driveable_areas('MIA')[0]

    def is_point_inside_roi(self, point, min_boundary, max_boundary):
        boundary_side = 0

        if point[0] < min_boundary:
            boundary_side = 11
        elif point[0] > max_boundary:
            boundary_side = 12

        if point[1] < min_boundary:
            boundary_side += 13
        elif point[1] > max_boundary:
            boundary_side += 14

        is_inside = True if boundary_side == 0 else False
        return is_inside, boundary_side

    def cut_polygon_by_roi(self, polygon):

        its_broken = False

        min_boundary = -100
        max_boundary = 100

        new_polygon = []

        _, prev_outside_side = self.is_point_inside_roi(polygon[-1], min_boundary, max_boundary)

        for i in range(len(polygon)):
            point = polygon[i]

            prev_point = polygon[(i + len(polygon) - 1) % len(polygon)]
            next_point = polygon[(i + 1) % len(polygon)]

            is_point_inside_roi, outside_side = self.is_point_inside_roi(point, min_boundary, max_boundary)

            is_next_point_inside_roi = min_boundary < next_point[0] < max_boundary and min_boundary < next_point[1] < max_boundary # todo: replace by is_point_inside_roi method
            is_prev_point_inside_roi = min_boundary < prev_point[0] < max_boundary and min_boundary < prev_point[1] < max_boundary # todo: replace by is_point_inside_roi method

            new_x = max(min(point[0], max_boundary), min_boundary)
            new_y = max(min(point[1], max_boundary), min_boundary)

            is_point_came_to_roi_corner = outside_side != 0 and prev_outside_side != 0 and outside_side > 20 and prev_outside_side < 15

            if is_point_inside_roi or is_next_point_inside_roi or is_prev_point_inside_roi or is_point_came_to_roi_corner:
                new_polygon.append([new_x, new_y])
            else:
                is_point_skip_corner = (10 < prev_outside_side <= 12 and 14 >= outside_side >= 13) or (10 < outside_side <= 12 and 14 >= prev_outside_side >= 13)

                if is_point_skip_corner:

                    # TODO: Debug it more. Plus refactoring!
                    prev_x = max(min(prev_point[0], max_boundary), min_boundary)
                    prev_y = max(min(prev_point[1], max_boundary), min_boundary)
                    new_polygon.append([prev_x, prev_y])
                    new_polygon.append([new_x, new_y])

                    #print('ITS BWOKEN')

                # if is_point_skip_corner:
                #     print('ITS BWOKEN')

            prev_outside_side = outside_side

            # if is_point_inside_roi or is_next_point_inside_roi or is_prev_point_inside_roi:
            #     new_point = np.array([new_x, new_y])
            #     new_polygon.append(new_point)

        return np.array(new_polygon)

    def draw_polygon(self, ax, polygon, cut_polygon, is_outline_polygon):

        if is_outline_polygon:
            color_out_poly = 'palegreen'
            color_poly = 'lightgray'
            color_boundary = 'g'
        else:
            color_out_poly = 'skyblue'
            color_poly = 'gray'
            color_boundary = 'y'

        # p = plotpoly(polygon, facecolor=color_out_poly)
        # ax.add_patch(p)

        p = plotpoly(cut_polygon, facecolor=color_poly)
        ax.add_patch(p)

        ax.plot(polygon[:, 0], polygon[:, 1], color='r', marker='o', markersize=6, linewidth=4)
        ax.plot([polygon[0][0], polygon[-1][0]], [polygon[0][1], polygon[-1][1]], color='r', marker='o', markersize=6, linewidth=4)

        ax.plot(cut_polygon[:, 0], cut_polygon[:, 1], color=color_boundary, marker='o', markersize=4)
        ax.plot([cut_polygon[0][0], cut_polygon[-1][0]], [cut_polygon[0][1], cut_polygon[-1][1]], color=color_boundary, marker='o', markersize=4)

        # for i in range(len(polygon)):
        #     plt.text(polygon[i][0] + 3, polygon[i][1] + 3, f'{i}')

        ax.set_xlim([-110, 110])
        ax.set_ylim([-110, 110])

        #plt.show()

    def out_of_drivable_area_check(self, da_polygons, pred_trajectory):  # TODO: Add visualization flag

        # start_time = time.time()

        points_outside_da_mask = np.zeros(len(pred_trajectory), dtype=bool)

        #fig, ax = plt.subplots()

        for i in range(len(da_polygons)):
            polygon = da_polygons[i]

            # start_item_time = time.time()

            # Exception for first polygon
            # Bad way to compare polygons, but otherwise we need to rotate outline polygons each time
            is_outline_polygon = len(polygon) == len(self.drivable_area_outline_pit) or len(polygon) == len(self.drivable_area_outline_mia)

            cropped_polygon = self.cut_polygon_by_roi(polygon)
            #print(f'replace poly: {len(polygon)} by {len(new_polygon)}')

            if len(cropped_polygon) < 3: # that basically means that polygon is out of ROI
                continue

            #self.draw_polygon(ax, polygon, new_polygon, is_outline_polygon)

            polygon = cropped_polygon

            polygon_obj = Polygon(polygon)
            current_pos = np.zeros(2)

            for idx, point in enumerate(pred_trajectory):
                current_pos = current_pos + point # cumulate current position
                point_obj = Point(current_pos)

                #contains_start_time = time.time()
                is_point_inside_polygon = polygon_obj.contains(point_obj)
                # print(f'time to calc contain : {time.time() - contains_start_time}')

                if (is_point_inside_polygon and not is_outline_polygon) or (not is_point_inside_polygon and is_outline_polygon):  # everything inside outline polygon is drivable area, for others polygons otherwise
                    points_outside_da_mask[idx] = True

            #print(f'Time per polygon: {time.time() - start_item_time}; amount of pt: {len(polygon)}')

        #
        # print(f'mask calc time :{time.time() - start_time}')
        # print()

        #plt.show()

        return points_outside_da_mask

    def polygon_normalization(self, drivable_areas_boundaries, orig_pos, rot):
        norm_drivable_areas_boundaries = []
        for da_boundary in drivable_areas_boundaries:
            da_boundary = da_boundary[:, :2]  # remove info about height

            rotated_boundary = np.matmul(rot, (da_boundary - orig_pos.reshape(-1, 2)).T).T
            norm_drivable_areas_boundaries.append(rotated_boundary)
        norm_drivable_areas_boundaries = np.array(norm_drivable_areas_boundaries, dtype=object)

        return norm_drivable_areas_boundaries

    def get_polylines_from_data(self, data):
        nodes = data.x.detach().numpy()
        cluster = data.cluster.detach().numpy()

        polylines = []
        for cluster_idc in np.unique(cluster):
            [indices] = np.where(cluster == cluster_idc)
            polyline = nodes[indices]

            polylines.append(polyline)

        return polylines

    def forward(self, data, pred_trajs, gt_trajs): # TODO: Create flag with/without visualization. Also: if data passed through network visualization brake
        #start_calc_time = time.time()

        data_cpu = data.cpu()

        polylines = self.get_polylines_from_data(data_cpu)

        # TODO: Get all polygons from map API and identify if point inside or outside of it

        query_search_range_manhattan = 100 # = radius # move to constructor

        orig_poses = data_cpu.orig.cpu().numpy()
        rotations = data_cpu.rot.cpu().numpy()
        cities = data_cpu.city_id.numpy()

        #print(f'Time finish .cpu(): {time.time() - start_calc_time}')

        outside_da_masks = []

        for i, pos in enumerate(orig_poses): # foreach each sample in batch
            #start_time = time.time()

            query_x = pos[0] # replace by single line
            query_y = pos[1]

            rot = rotations[i]
            pred = pred_trajs[i]
            gt = gt_trajs[i]

            city_id = cities[i]
            city_name = 'PIT' if city_id == 0 else 'MIA'

            query_min_x = query_x - query_search_range_manhattan
            query_max_x = query_x + query_search_range_manhattan
            query_min_y = query_y - query_search_range_manhattan
            query_max_y = query_y + query_search_range_manhattan

            #find_da_start_time = time.time()

            drivable_areas_boundaries = self.am.find_local_driveable_areas([query_min_x, query_max_x, query_min_y, query_max_y], city_name)

            #da_norm_time = time.time()
            norm_drivable_areas_boundaries = self.polygon_normalization(drivable_areas_boundaries, pos, rot)

            reshape_pred = pred.cpu().detach().numpy().reshape(30, 2)
            reshape_gt = gt.cpu().numpy().reshape(30, 2)

            # visual.draw_scene(polylines, reshape_gt, reshape_pred)
            # visual.draw_drivable_area(norm_drivable_areas_boundaries, -100, 100, -100, 100)

            #out_of_da_time = time.time()
            outside_da_mask = self.out_of_drivable_area_check(norm_drivable_areas_boundaries, reshape_pred)

            # Works only if batch size = 1 and data not forwarded into VectorNet
            # visual.draw_scene(polylines, reshape_gt, reshape_pred, outside_da_mask)
            # visual.draw_drivable_area(norm_drivable_areas_boundaries, -100, 100, -100, 100)

            outside_da_mask = np.repeat(outside_da_mask, 2) # back to shape 1x60 from 2x30
            outside_da_masks.append(outside_da_mask)

            #print(f'Step time: {time.time() - start_time}')
            #print()

        #print(f'Time finish main cycle: {time.time() - start_calc_time}')

        # get only points which outside driv area
        total_loss = 0
        for i, traj in enumerate(pred_trajs):
            mask = outside_da_masks[i]
            if not np.any(mask):
                continue

            new_traj = pred_trajs[i][mask]  # pick only points which outside drivable area
            new_gt = gt_trajs[i][mask]

            loss = F.mse_loss(new_traj, new_gt)
            total_loss += loss

        # total_calc_time = time.time() - start_calc_time
        # print(f'Total time: {total_calc_time}')

        return total_loss



