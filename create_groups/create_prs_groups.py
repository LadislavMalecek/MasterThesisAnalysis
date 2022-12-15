import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from create_groups.group_generator import GroupGenerator
from create_groups.dataset import Dataset
from create_groups.similarity_sampler_tools import SimilaritySamplerTools
import scipy.stats as ss

DESCRIPTION = 'Generates groups for a dataset based on PRS (probability respecting similarity) selection.'


# class SampledTopKGroupGenerator(GroupGenerator):
#     def generate_groups(
#             self,
#             dataset: Dataset,
#             output_dir: str,
#             group_size: int,
#             num_of_groups_to_generate: int,
#             number_of_candidates: int = 1000,
#     ) -> pd.DataFrame:
#         print(f"Generating top-k groups")
#         sim_sampler_tools = SimilaritySamplerTools(dataset)
#         groups = []
#         for _ in tqdm(range(num_of_groups_to_generate), total=num_of_groups_to_generate):
#             # draw one more which will become the pivot
#             candidates = sim_sampler_tools.draw_unique_candidates(number_of_candidates + 1)
#             pivot, candidates = candidates[0], candidates[1:]
#             # calculate the similarity of the pivot to all the candidates
#             candidates_similarities = sim_sampler_tools.calculate_similarity_of_candidates(pivot, candidates)
#             # sort the candidates by similarity
#             candidates_similarities = sorted(candidates_similarities, key=lambda x: x[1], reverse=True)
#             # take the top k candidates
#             top_k_candidates = [x[0] for x in candidates_similarities[:group_size - 1]]
#             group = [pivot] + top_k_candidates
#             groups.append(group)
#
#         df = pd.DataFrame(groups)
#         other_params = {'number_of_candidates': number_of_candidates}
#         self.save_group_df(
#             df,
#             output_dir=output_dir,
#             group_type='topk',
#             group_size=group_size,
#             other_params=other_params
#         )
#
#         return df


class PRSGroupGenerator(GroupGenerator):
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    @staticmethod
    def _calculate_weight(similarity, exp_params, scaling_exponent, scaling_width):
        scaling = scaling_exponent ** (similarity / scaling_width)
        prob_ratio = ss.expon.pdf(similarity, *exp_params)
        weight = scaling / prob_ratio
        return weight

    @staticmethod
    def _model_using_exp(samples) -> Tuple[float, float]:
        exp_params = ss.expon.fit(samples, method='MM')
        (e1, e2) = exp_params
        exp_params = (0, e2)
        return exp_params

    def _get_datasets_modeled_parameters(self, similarity_sampler_tools) -> Tuple[float, float]:
        similarity_samples = similarity_sampler_tools.get_similarity_sampling(num_of_samples=10_000, cache_dir=self.cache_dir)
        exp_params = self._model_using_exp(similarity_samples)
        print('exp_params', exp_params)
        return exp_params

    def _get_datasets_inter_quartile_range(self, similarity_sampler_tools):
        similarity_samples = similarity_sampler_tools.get_similarity_sampling(num_of_samples=10_000, cache_dir=self.cache_dir)
        similarity_samples.sort()

        first_quartile = similarity_samples[int(len(similarity_samples) * 0.25)]
        median = similarity_samples[int(len(similarity_samples) * 0.5)]
        third_quartile = similarity_samples[int(len(similarity_samples) * 0.75)]

        width_of_interquartile_range = third_quartile - first_quartile
        print(f'{first_quartile:.3f}, {median:.3f}, {third_quartile:.3f}, {width_of_interquartile_range:.5f}')
        return width_of_interquartile_range

    def generate_groups(
            self,
            dataset: Dataset,
            output_dir: str,
            group_size: int,
            num_of_groups_to_generate: int,
            scaling_exponent: float,
            scaling_width: float = None,
            number_of_candidates: int = 1000,
    ) -> pd.DataFrame:

        similarity_sampler_tools = SimilaritySamplerTools(dataset)
        exp_modeled_params = self._get_datasets_modeled_parameters(similarity_sampler_tools)

        if scaling_width is None:
            inter_quartile_range = self._get_datasets_inter_quartile_range(similarity_sampler_tools)
        else:
            inter_quartile_range = scaling_width

        print(f"Generating weighted groups")
        max_user_id = dataset.dataset_df['user_id'].max()
        groups_weighted = []
        groups_top_k = []
        for _ in tqdm(range(num_of_groups_to_generate), total=num_of_groups_to_generate):
            # draw one more which will become the pivot
            candidates = similarity_sampler_tools.draw_unique_candidates(number_of_candidates + 1)
            pivot, candidates = candidates[0], candidates[1:]
            assert len(candidates) == number_of_candidates
            # calculate the similarity of the pivot to all the candidates
            candidates_similarities = similarity_sampler_tools.calculate_similarity_of_candidates(pivot, candidates)

            candidates_similarities_w_weights = []
            for candidate_id, similarity in candidates_similarities:
                weight = self._calculate_weight(similarity, exp_modeled_params, scaling_exponent, scaling_width=inter_quartile_range)
                candidates_similarities_w_weights.append((candidate_id, similarity, weight))

            # take k candidates with respect to weights
            candidates = [x[0] for x in candidates_similarities_w_weights]
            weights = [x[2] for x in candidates_similarities_w_weights]
            weights = np.array(weights) / sum(weights)

            selected_weighted = list(np.random.choice(candidates, size=group_size - 1, p=weights, replace=False))

            # ordered_candidates = sorted(candidates_similarities_w_weights, key=lambda x: x[1], reverse=True)
            # if top_k:
            #     selected_top_k = ordered_candidates[:group_size - 1]
            #     selected_top_k = [x[0] for x in selected_top_k]

            #     group_top_k = [pivot] + selected_top_k
            #     assert len(group_top_k) == group_size
            #     groups_top_k.append(group_top_k)
            # assert len(selected_weighted) == len(selected_top_k)
            group_weighted = [pivot] + selected_weighted
            assert len(group_weighted) == group_size
            groups_weighted.append(group_weighted)

        # if top_k:
        #     df_top_k = pd.DataFrame(groups_top_k)
        #     save_group_df(df_top_k, type='top_k', dataset_name=dataset_name, group_size=group_size, other_params={'noc': number_of_candidates})

        df_weighted = pd.DataFrame(groups_weighted)
        other_params = {'se': scaling_exponent, 'noc': number_of_candidates}
        self.save_group_df(
            df_weighted,
            group_type='prs',
            output_dir=output_dir,
            group_size=group_size,
            other_params=other_params)
        return df_weighted


def parse_args():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--input', default='../datasets/kgrec/music_ratings.csv.gz', help='The dataset to use, needs to be csv dataframe with columns "user_id", "item_id" and optionally "rating".')
    parser.add_argument('--output-dir', default=None, help='The directory where the resulting matrices will be saved. Default is "groups" dir under the input data directory.')
    parser.add_argument('--cache-dir', default=None, help='The directory where the similarity sampling for dataset similarity distribution will be saved.')
    parser.add_argument('--group-sizes', default='4,6,8', type=str, help='Sizes of groups, numbers divided by a comma. Ex.: 5,6,7,8')
    parser.add_argument('--num-groups-to-generate', default=1000, type=int, help='Number of groups to generate.')
    parser.add_argument('--num-of-candidates', default=1000, type=int, help='Number of candidates from all users that will be drawn randomly from which to select the best k ones.')
    parser.add_argument('--scaling-exponent', default=1, type=int, help='Multiplying factor of the weighing for member sampling.')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input), 'groups')

    if args.cache_dir is None:
        args.cache_dir = os.path.join(args.output_dir, 'similarity_sampling_cache')

    args.group_sizes_list = list(map(lambda x: int(str.strip(x)), args.group_sizes.split(',')))

    return args


if __name__ == '__main__':
    args = parse_args()

    dataset = Dataset.load_dataset(path=args.input)
    for group_size in args.group_sizes_list:
        PRSGroupGenerator(cache_dir=args.cache_dir).generate_groups(
            dataset=dataset,
            output_dir=args.output_dir,
            group_size=group_size,
            num_of_groups_to_generate=args.num_groups_to_generate,
            scaling_exponent=args.scaling_exponent,
        )
