import os
from typing import List

import pandas as pd
import numpy as np


def load_mf_matrices(mf_path: str) -> tuple[np.ndarray, np.ndarray]:
    u_features = np.load(os.path.join(mf_path, 'U_features.npy'))
    i_features = np.load(os.path.join(mf_path, 'I_features.npy'))

    # implicit returns transposed matrices, fix it here
    if i_features.shape[0] < i_features.shape[1]:
        i_features = i_features.T
    if u_features.shape[0] < u_features.shape[1]:
        u_features = u_features.T

    print('U_features shape:', u_features.shape)
    print('I_features shape:', i_features.shape)
    return u_features, i_features


def load_groups(g_path: str) -> dict[str, pd.DataFrame]:
    group_file_names = [file for file in os.listdir(g_path) if file.endswith('.csv')]
    groups = {}
    for group_file in group_file_names:
        group_name = os.path.basename(group_file).split('.')[0]
        group_file_path = os.path.join(g_path, group_file)
        groups[group_name] = pd.read_csv(group_file_path, header=None)
    return groups

def calculate_dcg(ratings: np.ndarray) -> float:
    return np.sum(ratings / np.log2(np.arange(2, ratings.size + 2)))


def get_top_dict(all_metrics, sort_type_dict):
    all_metrics = all_metrics.copy().set_index('alg_name')
    all_metrics.drop(columns=['group_name', 'group_size'], inplace=True)
    sorted_by_score = {}
    for metric in all_metrics.columns:
        how_sort = sort_type_dict[metric]
        sorted_idx = all_metrics[metric].sort_values(ascending=(how_sort=='min')).index
        sorted_by_score[metric] = sorted_idx
    return sorted_by_score


class RatingsRetriever:
    def __init__(self, u_features: np.ndarray, i_features: np.ndarray):
        self.u_features = u_features
        self.i_features = i_features

        self.num_users = self.u_features.shape[0]
        self.num_items = self.i_features.shape[0]

        self._user_IDCGs = {}

    def get_ratings(self, user_ids: List[int], items: List[int]) -> List[float]:
        user_features = self.u_features[user_ids]
        item_features = self.i_features[items]
        # item_features = item_features.reshape(1, -1)
        return np.dot(user_features, item_features.T)

    def get_user_IDCG(self, user_id: int, size_top_k: int) -> float:
        if user_id in self._user_IDCGs:
            return self._user_IDCGs[user_id]
        
        ratings = self.get_ratings(user_id, list(range(self.num_items)))
        # get top k items
        top_n_ind = np.argpartition(ratings, -size_top_k)[-size_top_k:]
        sorted_top_k_items = np.sort(ratings[top_n_ind])[::-1]
        idcg = calculate_dcg(sorted_top_k_items)

        self._user_IDCGs[user_id] = idcg
        return idcg


def get_group_size_from_name(group_name):
    return int(group_name.split('_')[1])