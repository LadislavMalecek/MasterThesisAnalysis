import math
from typing import List

import numpy as np
from experiments.algorithms.commons import select_top_n_idx


class GreedyAlgorithms:
    @staticmethod
    def get_top_n_idx(group_items, n_candidates, exclude_idx: List[int] = []):
        top_candidates_ids_per_member = np.apply_along_axis(lambda u_items: select_top_n_idx(u_items, n_candidates, sort=False, exclude_idx=exclude_idx), 0, group_items)
        top_candidates_idx = np.array(sorted(set(top_candidates_ids_per_member.flatten())))
        return top_candidates_idx

    @staticmethod
    def avg_algorithm(group_items, top_n: int, n_candidates: int, member_weights=None, exclude_idx: List[int] = []):
        """
        Returns items ordered by average rating.
        """
        top_candidates_idx = GreedyAlgorithms.get_top_n_idx(group_items, n_candidates, exclude_idx=exclude_idx)
        candidate_group_items = group_items[top_candidates_idx, :]  # this is the first id mapping (to go back to original, index by top_candidates_idx)

        if member_weights is not None:
            candidate_group_items *= member_weights

        means = candidate_group_items.mean(axis=1)
        top_n_idx = select_top_n_idx(means, top_n)

        final_top_n_idx = top_candidates_idx[top_n_idx]
        return final_top_n_idx

    @staticmethod
    def lm_algorithm(group_items, top_n: int, n_candidates: int):
        """
        Returns items ordered by least min value across user rating.
        """
        top_candidates_idx = GreedyAlgorithms.get_top_n_idx(group_items, n_candidates)
        candidate_group_items = group_items[top_candidates_idx, :]  # this is the first id mapping (to go back to original, index by top_candidates_idx)

        mins = candidate_group_items.min(axis=1)
        top_n_idx = select_top_n_idx(mins, top_n)

        final_top_n_idx = top_candidates_idx[top_n_idx]
        return final_top_n_idx

    @staticmethod
    def fai_algorithm(group_items, top_n: int, n_candidates: int):
        """
        Returns items ordered by max of users each one by one per turn.
        So first item is selected as max of first user, second item by second and so on...
        """
        top_candidates_idx = GreedyAlgorithms.get_top_n_idx(group_items, n_candidates)
        candidate_group_items = group_items[top_candidates_idx, :]  # this is the first id mapping (to go back to original, index by top_candidates_idx)

        group_size = candidate_group_items.shape[1]
        # apply select_top_n to each user
        top_n_required_per_user = math.ceil(top_n / group_size)
        # we have candidate items, each item has is a list of scores for each group member
        # we get [[best item for u1, for u2, for u3, ..] [ second item for u1, ...]]
        top_n_idx_per_user = np.apply_along_axis(lambda row: select_top_n_idx(row, top_n_required_per_user), 0, candidate_group_items)
        # flatten the list to get the turn by turn top_n_idx
        top_n_idx = top_n_idx_per_user.flatten()[:top_n]

        final_top_n_idx = top_candidates_idx[top_n_idx]
        return final_top_n_idx

    @staticmethod
    def get_rank(score_list_np):
        """Best item has a rank of 1"""
        ranks = np.zeros(score_list_np.shape, dtype=np.int32)
        ranks[score_list_np.argsort()] = np.arange(start=score_list_np.shape[0], stop=0, step=-1)
        return ranks