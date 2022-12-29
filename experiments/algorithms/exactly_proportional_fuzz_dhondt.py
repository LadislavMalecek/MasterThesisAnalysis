from typing import List

import numpy as np

from experiments.algorithms.commons import select_top_n_idx


class EPFuzzDHondt:
    @staticmethod
    def run(group_items: np.array, top_n: int, n_candidates: int = None, member_weights: List[float] = None, exclude_idx: List[int] = []) -> np.array:
        group_size = group_items.shape[1]
        # print(group_size)

        if member_weights is None:
            member_weights = [1. / group_size] * group_size
        member_weights = np.array(member_weights)

        # print('Member weights: ', member_weights)
        # print('Member weights shape: ', member_weights.shape)

        top_candidates_ids_per_member = np.apply_along_axis(lambda u_items: select_top_n_idx(u_items, n_candidates, exclude_idx=exclude_idx), 0, group_items)
        # these are the original items ids
        top_candidates_idx = np.array(sorted(set(top_candidates_ids_per_member.flatten())))

        candidate_group_items = group_items[top_candidates_idx, :]  # this is the first id mapping (to go back to original, index by top_candidates_idx)

        # candidate_group_items = group_items # this is the first id mapping (to go back to original, index by top_candidates_idx)
        candidate_sum_utility = candidate_group_items.sum(axis=1)
        # print('Candidate sum utility: ', candidate_sum_utility)
        # print('Candidate sum utility shape: ', candidate_sum_utility.shape)

        total_user_utility_awarded = np.zeros(group_size)
        total_utility_awarded = 0.

        selected_items = []
        # top-n times select one item to the final list
        for i in range(top_n):
            # print()
            # print('Selecting item {}'.format(i))
            # print('Total utility awarded: ', total_utility_awarded)
            # print('Total user utility awarded: ', total_user_utility_awarded)

            prospected_total_utility = total_utility_awarded + candidate_sum_utility
            # print('Prospected total utility: ', prospected_total_utility)
            # print('Prospected total utility shape: ', prospected_total_utility.shape)

            # we need to stretch the dimension of the array to match the groups size, so that we can multiply it with the member weights
            stretched_prospected_total_utility = np.broadcast_to(np.expand_dims(prospected_total_utility, 1), (len(prospected_total_utility), group_size))
            allowed_utility_for_users = member_weights * stretched_prospected_total_utility
            unfulfilled_utility_for_users = np.maximum(0, allowed_utility_for_users - total_user_utility_awarded)
            # print('Unfulfilled utility for users: ', unfulfilled_utility_for_users)
            # print('Unfulfilled utility for users shape: ', unfulfilled_utility_for_users.shape)
            candidate_relevance = np.minimum(unfulfilled_utility_for_users, candidate_group_items)
            candidate_relevance = np.sum(candidate_relevance, axis=1)
            # print('Candidate relevance: ', candidate_relevance)
            # print('Candidate relevance shape: ', candidate_relevance.shape)

            # we are repeating the candidate selection with the already selected items
            # we therefore have to exclude the already selected items from the candidate selection
            idx_of_top_item = list(select_top_n_idx(candidate_relevance, 1, exclude_idx=selected_items))[0]
            selected_items.append(idx_of_top_item)

            total_user_utility_awarded += candidate_group_items[idx_of_top_item]
            total_utility_awarded += candidate_group_items[idx_of_top_item].sum()

        # now we need to get the original item ids from the final_candidates list and then top_candidates_idx
        final_top_candidates = top_candidates_idx[selected_items]
        return final_top_candidates