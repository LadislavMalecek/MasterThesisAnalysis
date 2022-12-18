import sys
from os import path
sys.path.append(path.join(sys.path[0], '..'))

import os

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import pandas as pd
from typing import List

import numpy as np
from tqdm import tqdm


from experiments.algorithms.GFAR import GFAR
from experiments.algorithms.XPO import XPO
from experiments.algorithms.dhondt_direct_optimize import DHondtDirectOptimize
from experiments.algorithms.exactly_proportional_fuzz_dhondt import EPFuzzDHondt

from experiments.algorithms.greedy_algorithms import GreedyAlgorithms


def get_items_for_user(user_id, u_features, i_features):
    items_ratings = u_features[:, user_id] @ i_features
    items_ids_w_ratings = [(item_id, rating) for item_id, rating in enumerate(items_ratings)]
    items_ids_w_ratings.sort(key=lambda x: x[1], reverse=True)
    return items_ids_w_ratings


def get_items_for_users(users_id: List, i_features, u_features):
    items_ratings = i_features @ u_features.T[:, users_id]
    # items_ratings = np.minimum(5, np.maximum(0, i_features.T @ u_features[:, users_id]))
    return items_ratings


def load_mf_matrices(path: str):
    u_features = np.load(os.path.join(path, 'U_features.npy'))
    i_features = np.load(os.path.join(path, 'I_features.npy'))
    print('U_features shape:', u_features.shape)
    print('I_features shape:', i_features.shape)
    return u_features, i_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-groups-directory', default='./datasets/kgrec/groups/',
                        help='The dataset to use, needs to be csv dataframe with columns "user_id", "item_id" and optionally "rating".')
    parser.add_argument('--input-mf', default='./datasets/kgrec/mf/', help='The dataset to use, needs to be csv dataframe with columns "user_id", "item_id" and optionally "rating".')
    parser.add_argument('--output-dir', default=None, help='The directory where the resulting matrices will be saved. Default is "groups" dir under the input data directory.')

    args = parser.parse_args()

    if args.output_dir is None:
        parent_path_groups = Path(args.input_groups_directory).parent
        args.output_dir = os.path.join(parent_path_groups, 'experiment_results')

    print(args)

    return args

def process_single_group(group_members):
    items = get_items_for_users(group_members, i_features=i_features, u_features=u_features)

    # avg_algorithm
    top_n_items_avg = GreedyAlgorithms.avg_algorithm(items, top_n=10, n_candidates=1000)
    results['avg'].append(top_n_items_avg)

    # lm_algorithm
    top_n_items_lm = GreedyAlgorithms.lm_algorithm(items, top_n=10, n_candidates=1000)
    results['lm'].append(top_n_items_lm)

    # fai_algorithm
    top_n_items_fai = GreedyAlgorithms.fai_algorithm(items, top_n=10, n_candidates=1000)
    results['fai'].append(top_n_items_fai)

    # xpo (npo) algorithm
    top_n_items_xpo = XPO.run(items, top_n=10, n_candidates=30, algo_type='XPO', mc_trials=100)
    results['xpo'].append(top_n_items_xpo)

    top_n_items_npo = XPO.run(items, top_n=10, n_candidates=30, algo_type='NPO', mc_trials=100)
    results['npo'].append(top_n_items_npo)

    # gfar algorithm
    top_n_items_gfar = GFAR.run(items, top_n=10, relevant_max_items=100, n_candidates=1000)
    results['gfar'].append(top_n_items_gfar)

    # dhonds algorithms
    top_n_ep_fuzz_dhondt = EPFuzzDHondt.run(items, top_n=10, n_candidates=1000)
    results['ep_fuzz_dhondt'].append(top_n_ep_fuzz_dhondt)

    top_n_dhondt_do = DHondtDirectOptimize.run(items, top_n=10, n_candidates=1000)
    results['dhondt_do'].append(top_n_dhondt_do)


if __name__ == '__main__':
    args = parse_args()

    # load groups

    for group_file in os.listdir(args.input_groups_directory):
        group_file = os.path.join(args.input_groups_directory, group_file)
        print(group_file)
        if not group_file.endswith('.csv'):
            print(f'Skipping {group_file}')
            continue

        groups = pd.read_csv(group_file, header=None)
        group_size = len(groups.columns)
        # concatenate first 5 columns to array of ints
        groups = groups.iloc[:, :group_size].values

        u_features, i_features = load_mf_matrices(args.input_mf)

        results = defaultdict(list)

        for group_members in tqdm(groups[0:10]):
            process_single_group(group_members)

        group_name = os.path.basename(group_file).split('.')[0]
        # save the results
        output_dir = os.path.join(args.output_dir, group_name)
        for alg_name, data in results.items():
            os.makedirs(output_dir, exist_ok=True)
            file_name = os.path.join(output_dir, f'{alg_name}.npy',)
            np.save(file_name, data)
