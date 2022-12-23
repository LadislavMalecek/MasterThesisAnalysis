from typing import List

import numpy as np

from experiments.algorithms.commons import select_top_n_idx


class DHondtDirectOptimize:
    @staticmethod
    def run(group_items, top_n, n_candidates=1000, member_weights=None) -> List[int]:
        group_size = group_items.shape[1]
        if member_weights is None:
            # will be normalized to 1 in the next step anyway, we can skip it here
            member_weights = [1] * group_size
        starting_voting_support = np.array(member_weights) / sum(member_weights)

        top_candidates_ids_per_member = np.apply_along_axis(lambda u_items: select_top_n_idx(u_items, n_candidates), 0, group_items)
        # these are the original items ids
        top_candidates_idx = np.array(sorted(set(top_candidates_ids_per_member.flatten())))
        candidate_group_items = group_items[top_candidates_idx, :] # this is the first id mapping (to go back to original, index by top_candidates_idx)

        # candidate_group_items = group_items # this is the first id mapping (to go back to original, index by top_candidates_idx)

        current_voting_support = starting_voting_support.copy()
        selected_items_relevance = np.zeros(group_size)
        selected_items = []
        # top-n times select one item to the final list
        for i in range(top_n):
            candidate_relevance = np.sum(candidate_group_items * current_voting_support, axis=1)
            idx_of_top_item = list(select_top_n_idx(candidate_relevance, 1, exclude_idx=selected_items))[0]

            selected_items.append(idx_of_top_item)
            selected_items_relevance += candidate_group_items[idx_of_top_item]
            current_voting_support = starting_voting_support / selected_items_relevance

        # now we need to get the original item ids from the final_candidates list and then top_candidates_idx
        final_top_candidates = top_candidates_idx[selected_items]
        return final_top_candidates