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
    parser.add_argument('--num-longterm-runs', default=5, type=int, help='The number of longterm runs to perform.')
    args = parser.parse_args()

    if args.output_dir is None:
        parent_path_groups = Path(args.input_groups_directory).parent
        args.output_dir = os.path.join(parent_path_groups, 'experiment_results/longterm')

    print(args)

    return args

def get_weights_for_lt(items, selected_top_idx, num_members, runs_so_far):
    # FROM UMAP 2021 - Weights of each user are defined as a non-negative difference between
    # the exactly proportional share of total relevance after the recommendation
    # session and the actual sum of relevance scores gained by the user so far.

    # meaning how much he could have gotten - how much he got
    # how much he could have gotten is just the sum of all scores for all recommended items and all members / number of members
    # x = eps of total relevance - actual sum of relevance scores
    # max from 0 and x
    item_ratings = items[selected_top_idx, :]
    total_allowed_relevance = np.sum(item_ratings)
    total_allowed_relevance_with_next_run = total_allowed_relevance / runs_so_far * (runs_so_far + 1)
    total_allowed_relevance_per_user = total_allowed_relevance_with_next_run / num_members
    total_received_relevance_per_member = np.sum(item_ratings, axis=0)
    weight = np.maximum(0, total_allowed_relevance_per_user - total_received_relevance_per_member)
    return weight / np.sum(weight)


def process_single_group_lt(results, group_members, num_longterm_runs):
    items = get_items_for_users(group_members, i_features=i_features, u_features=u_features)

    # avg_uniform_algorithm
    top_n_items_avg_uniform = GreedyAlgorithms.avg_algorithm(items, top_n=50, n_candidates=1000)
    results['avg_uniform'].append(top_n_items_avg_uniform)

    weight_trace = {}

    # dhonds algorithms
    weights = np.ones(len(group_members)) / len(group_members)
    top_n_avg = []
    lt_weights_avg = []
    for run_i in range(num_longterm_runs):
        top_n_avg.extend(GreedyAlgorithms.avg_algorithm(
            items,
            top_n=10,
            n_candidates=1000,
            member_weights=weights,
            exclude_idx=top_n_avg))
        weights = get_weights_for_lt(items, top_n_avg, len(group_members), runs_so_far=run_i + 1)
        lt_weights_avg.append(weights)
    results['avg'].append(top_n_avg)
    weight_trace['avg'] = lt_weights_avg

    # dhonds algorithms
    weights = np.ones(len(group_members)) / len(group_members)
    top_n_ep_fuzz_dhondt = []
    lt_weights_ep_fuzz_dhondt = []
    for run_i in range(num_longterm_runs):
        top_n_ep_fuzz_dhondt.extend(EPFuzzDHondt.run(
            items, 
            top_n=10, 
            n_candidates=1000, 
            member_weights=weights, 
            exclude_idx=top_n_ep_fuzz_dhondt))
        weights = get_weights_for_lt(items, top_n_ep_fuzz_dhondt, len(group_members), runs_so_far=run_i + 1)
        lt_weights_ep_fuzz_dhondt.append(weights)
    results['ep_fuzz_dhondt'].append(top_n_ep_fuzz_dhondt)
    weight_trace['ep_fuzz_dhondt'] = lt_weights_ep_fuzz_dhondt

    
    weights = np.ones(len(group_members)) / len(group_members)
    top_n_dhondt_do = []
    lt_weights_dhondt_do = []
    for run_i in range(num_longterm_runs):
        top_n_dhondt_do.extend(DHondtDirectOptimize.run(items, top_n=10, n_candidates=1000, member_weights=weights))
        weights = get_weights_for_lt(items, top_n_dhondt_do, len(group_members), runs_so_far=run_i + 1)
        lt_weights_dhondt_do.append(weights)
    results['dhondt_do'].append(top_n_dhondt_do)
    weight_trace['dhondt_do'] = lt_weights_dhondt_do
    
    return weight_trace


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
        
        # concatenate first 5 columns to array of ints
        groups = groups.iloc[:, :group_size].values

        results = defaultdict(list)
        weight_trace = defaultdict(list)

        for group_members in tqdm(list(groups)):
            weights_trace = process_single_group_lt(results, group_members, num_longterm_runs=args.num_longterm_runs)
            for alg_name, data in weights_trace.items():
                weight_trace[alg_name].append(data)

        group_name = os.path.basename(group_file).split('.')[0]
        # save the results
        output_dir = os.path.join(args.output_dir, group_name)
        for alg_name, data in results.items():
            os.makedirs(output_dir, exist_ok=True)
            file_name = os.path.join(output_dir, f'{alg_name}.npy', )
            print('saving results to:' + file_name)
            np.save(file_name, data)

        trace_dir = os.path.join(output_dir, 'weight_traces')
        os.makedirs(trace_dir, exist_ok=True)
        print('saving weight traces to:' + trace_dir)
        for alg_name, data in weight_trace.items():
            trace_file_name = os.path.join(trace_dir, f'{alg_name}_trace.npy', )
            np.save(trace_file_name, data)
