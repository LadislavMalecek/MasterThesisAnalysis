import numpy as np
from scipy.stats import rankdata


def select_top_n_idx(score_list, top_n, top='max', sort=True, exclude_idx=[]):
    if top != 'max' and top != 'min':
        raise ValueError('top must be either Max or Min')
    if top == 'max':
        score_list = -score_list

    select_top_n = top_n + len(exclude_idx)
    top_n_ind = np.argpartition(score_list, select_top_n)[:select_top_n]

    if sort:
        top_n_ind = top_n_ind[np.argsort(score_list[top_n_ind])]

    if exclude_idx:
        top_n_ind = [idx for idx in top_n_ind if idx not in exclude_idx]
    return top_n_ind[0:top_n]


# borda count that is limited only to top-max_rel_items, if you are not in the top-max_rel_items, you get 0
def get_borda_rel(candidate_group_items_np, max_rel_items):
    rel_idx = select_top_n_idx(candidate_group_items_np, max_rel_items, top='max', sort=False)
    # print(candidate_group_items_np[rel_idx])
    x = candidate_group_items_np[rel_idx]
    rel_borda = rankdata(-candidate_group_items_np[rel_idx], method='max')
    # print(rel_borda)

    rel_all = np.zeros(len(candidate_group_items_np))
    rel_all[rel_idx] = rel_borda
    return rel_all