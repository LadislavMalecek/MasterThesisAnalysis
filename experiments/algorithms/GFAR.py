from typing import List

import numpy as np

from experiments.algorithms.commons import select_top_n_idx, get_borda_rel


class GFAR:
    @staticmethod
    def run(group_items, top_n: int, relevant_max_items: int, n_candidates: int) -> List[int]:
        group_size = group_items.shape[1]

        top_candidates_ids_per_member = np.apply_along_axis(lambda u_items: select_top_n_idx(u_items, n_candidates, sort=False), 0, group_items)
        # these are the original items ids
        top_candidates_idx = np.array(sorted(set(top_candidates_ids_per_member.flatten())))
        # get the candidate group items for each member
        candidate_group_items = group_items[top_candidates_idx, :]  # this is the first id mapping (to go back to original, index by top_candidates_idx)

        borda_rel_of_candidates = np.apply_along_axis(lambda items_for_user: get_borda_rel(items_for_user, relevant_max_items), 0, candidate_group_items)
        total_relevance_for_users = borda_rel_of_candidates.sum(axis=0)
        p_relevant = borda_rel_of_candidates / total_relevance_for_users

        selected_items = []
        # this is the inside of the product in calculating the relevance for set of selected
        prob_selected_not_relevant = np.ones(group_size)

        # top-n times select one item to the final list
        for i in range(top_n):
            marginal_gain = p_relevant * prob_selected_not_relevant
            item_marginal_gain = marginal_gain.sum(axis=1)
            # select the item with the highest marginal gain
            item_id = list(select_top_n_idx(item_marginal_gain, 1, exclude_idx=selected_items))[0]
            selected_items.append(item_id)

            # update the probability of selected items not being relevant
            prob_selected_not_relevant *= (1 - p_relevant[item_id])

        # now we need to get the original item ids from the final_candidates list and then top_candidates_idx
        final_top_candidates = top_candidates_idx[selected_items]
        return final_top_candidates