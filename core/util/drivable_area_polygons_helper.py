from enum import IntFlag
import numpy as np

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as plotpoly


class BOUNDARY_SIDE(IntFlag):
    NONE = 0
    LEFT = 11,
    RIGHT = 12,
    DOWN = 13
    UP = 14,


def is_point_inside_roi(point, min_boundary, max_boundary):
    clamped_side = BOUNDARY_SIDE.NONE

    if point[0] < min_boundary:
        clamped_side = clamped_side + BOUNDARY_SIDE.LEFT
    elif point[0] > max_boundary:
        clamped_side = clamped_side + BOUNDARY_SIDE.RIGHT

    if point[1] < min_boundary:
        clamped_side = clamped_side + BOUNDARY_SIDE.DOWN
    elif point[1] > max_boundary:
        clamped_side = clamped_side + BOUNDARY_SIDE.UP

    is_inside = True if clamped_side == BOUNDARY_SIDE.NONE else False
    return is_inside, clamped_side


def is_point_clamped_by_corner(prev_clamped_side, new_clamped_side):
    prev_clamped_side = int(prev_clamped_side)
    new_clamped_side = int(new_clamped_side)

    # if point in come corner (UP-LEFT, UP-RIGHT, DOWN-LEFT, DOWN-RIGHT)
    return new_clamped_side != BOUNDARY_SIDE.NONE and prev_clamped_side != BOUNDARY_SIDE.NONE and new_clamped_side > 20 and prev_clamped_side < 15


def is_point_skip_corner(prev_clamped_side, new_clamped_side):
    prev_clamped_side = int(prev_clamped_side)
    new_clamped_side = int(new_clamped_side)

    # if prev clamped side outside of horizontal or vertical boundary (not both!) and next outside of vertical or horizontal
    # Example: move from LEFT-side to UP-side without being in corner
    return (10 < prev_clamped_side <= 12 and 14 >= new_clamped_side >= 13) or (10 < new_clamped_side <= 12 and 14 >= prev_clamped_side >= 13)


def clamp_point(point, min_boundary, max_boundary):
    new_x = max(min(point[0], max_boundary), min_boundary)
    new_y = max(min(point[1], max_boundary), min_boundary)

    return [new_x, new_y]


def cut_polygon_by_roi(polygon):

    min_boundary = -100
    max_boundary = 100

    clamped_polygon = []

    for i in range(len(polygon)):
        point = polygon[i]

        prev_point = polygon[(i + len(polygon) - 1) % len(polygon)]
        next_point = polygon[(i + 1) % len(polygon)]

        is_pt_inside_roi, clamped_side = is_point_inside_roi(point, min_boundary, max_boundary)

        is_prev_pt_inside_roi, prev_clamped_side = is_point_inside_roi(prev_point, min_boundary, max_boundary)
        is_next_pt_inside_roi, _ = is_point_inside_roi(next_point, min_boundary, max_boundary)

        clamped_point = clamp_point(point, min_boundary, max_boundary)

        is_point_clamped_in_corner = is_point_clamped_by_corner(prev_clamped_side, clamped_side)

        if is_pt_inside_roi or is_prev_pt_inside_roi or is_next_pt_inside_roi or is_point_clamped_in_corner:
            clamped_polygon.append(clamped_point)
        else:
            is_pt_skip_corner = is_point_skip_corner(prev_clamped_side, clamped_side)

            if is_pt_skip_corner:
                prev_clamped_point = clamp_point(prev_point, min_boundary, max_boundary)

                clamped_polygon.append(prev_clamped_point)
                clamped_polygon.append(clamped_point)

    return np.array(clamped_polygon)


def draw_polygon(ax, polygon, cut_polygon, is_outline_polygon):
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

    # ax.plot(polygon[:, 0], polygon[:, 1], color='r', marker='o', markersize=6, linewidth=4)
    # ax.plot([polygon[0][0], polygon[-1][0]], [polygon[0][1], polygon[-1][1]], color='r', marker='o', markersize=6, linewidth=4)

    ax.plot(cut_polygon[:, 0], cut_polygon[:, 1], color=color_boundary, marker='o', markersize=4)
    ax.plot([cut_polygon[0][0], cut_polygon[-1][0]], [cut_polygon[0][1], cut_polygon[-1][1]], color=color_boundary, marker='o', markersize=4)

    # for i in range(len(polygon)):
    #     plt.text(polygon[i][0] + 3, polygon[i][1] + 3, f'{i}')

    ax.set_xlim([-110, 110])
    ax.set_ylim([-110, 110])


def out_of_drivable_area_check(da_polygons, outline_polygon_index, pred_trajectory, show_visualization=False):

    points_outside_da_mask = np.zeros(len(pred_trajectory), dtype=bool)

    if show_visualization:
        fig, ax = plt.subplots()

    for i in range(len(da_polygons)):
        cropped_polygon = da_polygons[i]

        # Exception for first polygon
        # Bad way to compare polygons, but otherwise we need to rotate outline polygons each time
        is_outline_polygon = True if i == outline_polygon_index else False

        if len(cropped_polygon) < 3:  # that basically means that polygon is out of ROI
            continue

        if show_visualization:
            draw_polygon(ax, None, cropped_polygon, is_outline_polygon)

        polygon_obj = Polygon(cropped_polygon)

        for idx, point in enumerate(pred_trajectory):
            if point[0] < -100 or point[0] > 100 or point[1] < -100 or point[1] > 100:
                continue

            point_obj = Point(point)

            is_point_inside_polygon = polygon_obj.contains(point_obj)

            if (is_point_inside_polygon and not is_outline_polygon) or (not is_point_inside_polygon and is_outline_polygon):  # everything inside outline polygon is drivable area, for others polygons otherwise
                points_outside_da_mask[idx] = True

    if show_visualization:
        plt.grid()
        ax.set_xlim([-110, 110])
        ax.set_ylim([-110, 110])
        plt.show()

    return points_outside_da_mask


def get_da_polylines_from_data(data):
    nodes = data.x.detach().numpy()
    cluster = data.cluster.detach().numpy()

    polylines = []
    for cluster_idc in np.unique(cluster):
        [indices] = np.where(cluster == cluster_idc)
        polyline = nodes[indices]

        polylines.append(polyline)

    return polylines