import numpy as np
from matplotlib import pyplot as plt

TARGET_COLOR = 'r'
OTHER_AGENT_COLOR = 'gray'

TARGET_HISTORY_COLOR = (191 / 255, 105 / 255, 105 / 255)
OTHER_AGENT_HISTORY_COLOR = (160 / 255, 160 / 255, 160 / 255)

AGENT_POINT_SIZE = 7


def is_trajectory_polyline(polyline):
    all_nodes_steps = polyline[:, 4]
    # return True if is a trajectory, False if is a line
    return not (len(polyline) > 0 and np.all((all_nodes_steps == 0)))


def draw_line(plt, line_vectors_centers, line_vectors):
    color = 'gray'
    line_str = (2.0 * line_vectors_centers - line_vectors) / 2.0
    line_end = (2.0 * line_vectors_centers[-1, :] + line_vectors[-1, :]) / 2.0
    line = np.vstack([line_str, line_end.reshape(-1, 2)])
    line_coords = list(zip(*line))
    plt.plot(line_coords[0], line_coords[1], "--", color=color, alpha=1, linewidth=1, zorder=0)


def draw_trajectory(plt, line_vectors_centers, line_vectors, is_target_drawn):
    color_current = OTHER_AGENT_COLOR if is_target_drawn else TARGET_COLOR
    color_history = OTHER_AGENT_HISTORY_COLOR if is_target_drawn else TARGET_HISTORY_COLOR

    z_order = 0 if is_target_drawn else 1

    for i in range(len(line_vectors_centers)):
        start_pos = line_vectors_centers[i]
        finish_pos = start_pos + line_vectors[i]

        plt.plot([start_pos[0], finish_pos[0]], [start_pos[1], finish_pos[1]], color=color_history, alpha=1, linewidth=3, zorder=z_order)

    # Circle as current position
    finish_pos = line_vectors_centers[len(line_vectors_centers) - 1] + line_vectors[len(line_vectors_centers) - 1]
    plt.plot(finish_pos[0], finish_pos[1], 'o', markersize=AGENT_POINT_SIZE, color=color_current, zorder=5)


def draw_future_trajectory(plt, trajectory, color, width):
    # Convert from array of offsets to point w.r.t start agent position
    trajectory = np.cumsum(trajectory, axis=0)
    plt.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=1, linewidth=width, zorder=2)


def draw_scene(polylines, future_trajectory, model_prediction):
    fig = plt.figure(0, figsize=(8, 7))
    fig.clear()

    is_target_drawn = False

    for polyline in polylines:
        line_vectors_centers = polyline[:, 0: 2]
        line_vectors = polyline[:, 2: 4]

        if is_trajectory_polyline(polyline):
            draw_trajectory(plt, line_vectors_centers, line_vectors, is_target_drawn)
            is_target_drawn = True
        else:
            draw_line(plt, line_vectors_centers, line_vectors)

    draw_future_trajectory(plt, future_trajectory, 'g', width=3)
    draw_future_trajectory(plt, model_prediction, 'yellow', width=2)

    plt.xlabel("Map X")
    plt.ylabel("Map Y")
    plt.axis("off")
    plt.show()