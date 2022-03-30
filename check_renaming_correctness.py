# TODO : Load 2 pickle files from different sources and compare

import pickle
import pandas as pd

filename = '/home/techtoker/projects/TNT-Trajectory-Predition/dataset/interm_data_small/train_intermediate/raw/features_43725.pkl'
new_naming_df = pd.read_pickle(filename)

filename = '/home/techtoker/projects/TNT-Trajectory-Predition/dataset/before_interm_data_small/train_intermediate/raw/features_43725.pkl'
old_naming_df = pd.read_pickle(filename)


old_naming_df = old_naming_df.rename(columns={
                                              'trajs': 'all_agents_trajectories',
                                              'steps': 'agents_timestamp_presence',
                                              'orig': 'origin_pos',
                                              'theta': 'rotation_angle',
                                              'rot': 'rotation_matrix',
                                              'feats': 'all_agents_history',
                                              'has_obss': 'agents_history_presence',
                                              'has_preds': 'agents_future_presence',
                                              'gt_preds': 'future_trajectories',
                                              'tar_candts': 'target_candidates',
                                              'gt_candts': 'target_candidates_onehot',
                                              'gt_tar_offset': 'target_offset_gt',
                                              'ref_ctr_lines': 'centerline_splines',
                                              'ref_cetr_idx': 'reference_centerline_idx',
                                              })

old_graph_keys = ['ctrs', 'feats', 'turn', 'control', 'intersect']
new_graph_keys = ['centers', 'lines_vectors', 'lines_turn_info', 'lines_traffic_control_info', 'lines_intersect_info']

print(old_graph_keys)
print(new_graph_keys)

for key, n_key in zip(old_graph_keys, new_graph_keys):
    print(f'old: {key}, new: {n_key}')
    old_naming_df.iloc[0]['graph'][n_key] = old_naming_df.iloc[0]['graph'].pop(key)

# print(old_naming_df.iloc[0]['graph'])
# print()
# print(new_naming_df.iloc[0]['graph'])

# print()
#
# for key in new_naming_df.iloc[0]['graph'].keys():
#     print(f'key: {key}')
#     print(new_naming_df.iloc[0]['graph'][key])
#     print()
#     print(old_naming_df.iloc[0]['graph'][key])
#     print()
#     print('/n')
#     print()


print()
for column in old_naming_df:
    print(column)
    print()
    print(old_naming_df[column].iloc[0])