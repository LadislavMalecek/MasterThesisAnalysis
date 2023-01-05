import sys
from os import path
sys.path.append(path.join(sys.path[0], '..'))

import os

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

import numpy as np
from tqdm import tqdm

from experiments.algorithms.dhondt_direct_optimize import DHondtDirectOptimize
from experiments.algorithms.exactly_proportional_fuzz_dhondt import EPFuzzDHondt

from experiments.algorithms.greedy_algorithms import GreedyAlgorithms

from experiments.run_uniform_algorithms import load_mf_matrices, get_items_for_users


def load_weights(input_groups_directory: str, group_size: int):
    weights_path = os.path.join(input_groups_directory, f'weights/group_weights_{group_size}.csv')
    return pd.read_csv(weights_path, header=None).to_numpy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-groups-directory', default='./datasets/kgrec/groups/',
                        help='The dataset to use, needs to be csv dataframe with columns "user_id", "item_id" and optionally "rating".')
    parser.add_argument('--input-mf', default='./datasets/kgrec/mf/', help='The dataset to use, needs to be csv dataframe with columns "user_id", "item_id" and optionally "rating".')
    parser.add_argument('--output-dir', default=None, help='The directory where the resulting matrices will be saved. Default is "groups" dir under the input data directory.')
    parser.add_argument('--group-sizes', default='3,4,6,8', type=str, help='Sizes of groups, numbers divided by a comma. Ex.: 5,6,7,8')

    args = parser.parse_args()

    if args.output_dir is None:
        parent_path_groups = Path(args.input_groups_directory).parent
        args.output_dir = os.path.join(parent_path_groups, 'experiment_results/weighted')

    args.group_sizes_set = set(map(lambda x: int(str.strip(x)), args.group_sizes.split(',')))

    print(args)

    return args


def process_single_group(results, group_members, group_weights):
    items = get_items_for_users(group_members, i_features=i_features, u_features=u_features)

    # avg_uniform_algorithm
    top_n_items_avg_uniform = GreedyAlgorithms.avg_algorithm(items, top_n=10, n_candidates=1000)
    results['avg_uniform'].append(top_n_items_avg_uniform)

    # avg_algorithm
    top_n_items_avg = GreedyAlgorithms.avg_algorithm(items, top_n=10, n_candidates=1000, member_weights=group_weights)
    results['avg'].append(top_n_items_avg)

    # dhonds algorithms
    top_n_ep_fuzz_dhondt = EPFuzzDHondt.run(items, top_n=10, n_candidates=1000, member_weights=group_weights)
    results['ep_fuzz_dhondt'].append(top_n_ep_fuzz_dhondt)

    top_n_dhondt_do = DHondtDirectOptimize.run(items, top_n=10, n_candidates=1000, member_weights=group_weights)
    results['dhondt_do'].append(top_n_dhondt_do)


if __name__ == '__main__':
    args = parse_args()


    u_features, i_features = load_mf_matrices(args.input_mf)

    # load groups
    for group_file in os.listdir(args.input_groups_directory):
        if group_file.startswith('random') or group_file.startswith('topk'):
            print(f'Skipping {group_file}')
            continue

        group_file = os.path.join(args.input_groups_directory, group_file)
        print(group_file)
        if not group_file.endswith('.csv'):
            print(f'Skipping {group_file}')
            continue

        groups = pd.read_csv(group_file, header=None)
        group_size = len(groups.columns)

        if group_size not in args.group_sizes_set:
            continue

        # concatenate first 5 columns to array of ints
        groups = groups.iloc[:, :group_size].values

        group_weights = load_weights(args.input_groups_directory, group_size)

        results = defaultdict(list)

        for group_members, group_weights in tqdm(list(zip(groups, group_weights))):
            process_single_group(results, group_members, group_weights)

        group_name = os.path.basename(group_file).split('.')[0]
        # save the results
        output_dir = os.path.join(args.output_dir, group_name)
        for alg_name, data in results.items():
            os.makedirs(output_dir, exist_ok=True)
            file_name = os.path.join(output_dir, f'{alg_name}.npy', )
            print('saving results to:' + file_name)
            np.save(file_name, data)
