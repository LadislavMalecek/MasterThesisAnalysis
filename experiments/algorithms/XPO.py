import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from experiments.algorithms.commons import select_top_n_idx


class XPO:
    @staticmethod
    def get_rank(candidate_group_items_np):
        return rankdata(-candidate_group_items_np, method='min')

    @staticmethod
    def run(group_items, top_n: int, n_candidates: int, algo_type: str = 'XPO', mc_trials: int = 1000) -> np.array:
        if algo_type != 'XPO' and algo_type != 'NPO':
            raise ValueError(f'Unknown type: {algo_type}, only XPO and NPO are supported')

        group_size = group_items.shape[1]
        # first select top candidates for each member
        top_candidates_ids_per_member = np.apply_along_axis(lambda u_items: select_top_n_idx(u_items, n_candidates, sort=False), 0, group_items)
        # these are the original items ids
        top_candidates_idx = np.array(sorted(set(top_candidates_ids_per_member.flatten())))

        # get the candidate group items for each member
        candidate_group_items = group_items[top_candidates_idx, :]  # this is the first id mapping (to go back to original, index by top_candidates_idx)

        # from now on, item ids are ids into this candidate group items list

        # calculate the rank of candidates and all values bigger than n_candidates are set to n_candidates + 1
        # ranks will therefore be [1, 31]
        rank_of_candidates = np.apply_along_axis(XPO.get_rank, 0, candidate_group_items)
        rank_of_candidates = np.minimum(rank_of_candidates, n_candidates + 1)

        # now we want to compare all pairs of items to calculate the their Pareto optimality level
        # we say that item a is pareto optimal over b if for all group members rank(a) <= rank(b)
        # they can be both pareto optimal(if their ranks are equal)
        # or non-pareto optimal(if their dominate each other for different group members)
        number_of_items = len(candidate_group_items)
        # print(f'Number of items: {number_of_items}')
        pareto_levels = [1] * number_of_items  # same ids as candidate_group_items
        for a, b in itertools.combinations(range(len(candidate_group_items)), 2):
            rank_of_candidates_a = rank_of_candidates[a]
            rank_of_candidates_b = rank_of_candidates[b]

            rank_dif = rank_of_candidates_a - rank_of_candidates_b
            # for each group member, if a is pareto optimal over b, then all rank_difs will be negative or zero
            # if b is pareto optimal over 1, then all rank_difs will be positive or zero
            # and atleast one rank for any group member will negative or positive (non-zero) respectively
            a_paretooptim = np.all(rank_dif <= 0) & np.any(rank_dif < 0)
            b_paretooptim = np.all(rank_dif >= 0) & np.any(rank_dif > 0)
            # the lower the level the better, as an item, if you are bester (dominated) we lower your level by 1
            # where level 1 is best and level infinity is worst
            if a_paretooptim:
                pareto_levels[b] += 1
            if b_paretooptim:
                pareto_levels[a] += 1

        # now group the items by pareto level
        pareto_levels_pd = pd.DataFrame(enumerate(pareto_levels), columns=['item_id', 'pareto_level'])
        pareto_levels_pd = pareto_levels_pd.sort_values(by='pareto_level', ascending=True)

        # select candidates by pareto level, cut off at pareto_level = top_n
        if algo_type == 'NPO':
            final_candidates = pareto_levels_pd[pareto_levels_pd['pareto_level'] <= top_n + 1]['item_id'].explode().to_numpy()

        # select candidates starting at pareto level 1 and cut off when the number of candidates is equal or greater than top_n
        if algo_type == 'XPO':
            # add cumulative count of items to the grouped dataframe
            pareto_levels_grouped = pareto_levels_pd.groupby('pareto_level').agg(level=('pareto_level', 'first'), items=('item_id', 'unique'), items_count=('item_id', 'count'))
            pareto_levels_grouped['cum_items_count'] = pareto_levels_grouped['items_count'].cumsum()
            idx_of_first_larger_than_top_k = (pareto_levels_grouped['cum_items_count'] > top_n + 1).idxmax()
            # select all pareto levels that will get us top_n items and explode the lists to get the top_n items
            final_candidates = pareto_levels_grouped.iloc[0:idx_of_first_larger_than_top_k]['items'].explode().to_numpy(dtype=np.int64)

        final_candidates_group_items = group_items[final_candidates, :]  # second id mapping (to go back index by final_candidates)

        # simple Monte Carlo method for computing probabilities of linear aggregation strategy ratios (analytical solution not feasible)
        # now we want to try many different weights of users and award points to winners based on the weighted ranks

        mc_score = np.zeros(len(final_candidates))

        for i in range(mc_trials):
            # get random weights summing up to 1 for the group members
            weights = np.random.random(group_size)
            weights /= weights.sum()

            group_items_score = np.sum(weights * final_candidates_group_items, axis=1)
            top_n_idx = select_top_n_idx(group_items_score, top_n, sort=False)
            mc_score[top_n_idx] += 1

        # print(f'MC score: {mc_score}')
        top_n_idx = select_top_n_idx(mc_score, top_n)
        # print(top_n_idx)
        # print(final_candidates[top_n_idx])
        # now we need to get the original item ids from the final_candidates list and then top_candidates_idx
        final_top_candidates = top_candidates_idx[final_candidates[top_n_idx]]
        # print(f'Final top candidates: {final_top_candidates}')
        return final_top_candidates