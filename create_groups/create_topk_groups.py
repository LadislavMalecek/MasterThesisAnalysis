import argparse
import os

import pandas as pd
from tqdm import tqdm

from create_groups.group_generator import GroupGenerator
from create_groups.dataset import Dataset
from create_groups.similarity_sampler_tools import SimilaritySamplerTools

DESCRIPTION = 'Generates groups for a dataset based on the most similar users from a candidate list.'


class SampledTopKGroupGenerator(GroupGenerator):
    def generate_groups(
            self,
            dataset: Dataset,
            output_dir: str,
            group_size: int,
            num_of_groups_to_generate: int,
            number_of_candidates: int = 1000,
    ) -> pd.DataFrame:
        print(f"Generating top-k groups")
        sim_sampler_tools = SimilaritySamplerTools(dataset)
        groups = []
        for _ in tqdm(range(num_of_groups_to_generate), total=num_of_groups_to_generate):
            # draw one more which will become the pivot
            candidates = sim_sampler_tools.draw_unique_candidates(number_of_candidates + 1)
            pivot, candidates = candidates[0], candidates[1:]
            # calculate the similarity of the pivot to all the candidates
            candidates_similarities = sim_sampler_tools.calculate_similarity_of_candidates(pivot, candidates)
            # sort the candidates by similarity
            candidates_similarities = sorted(candidates_similarities, key=lambda x: x[1], reverse=True)
            # take the top k candidates
            top_k_candidates = [x[0] for x in candidates_similarities[:group_size - 1]]
            group = [pivot] + top_k_candidates
            groups.append(group)

        df = pd.DataFrame(groups)
        other_params = {'noc': number_of_candidates}
        self.save_group_df(
            df,
            output_dir=output_dir,
            group_type='topk',
            group_size=group_size,
            other_params=other_params
        )

        return df


def parse_args():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--input', default='../datasets/kgrec/music_ratings.csv.gz', help='The dataset to use, needs to be csv dataframe with columns "user_id", "item_id" and optionally "rating".')
    parser.add_argument('--output-dir', default=None, help='The directory where the resulting matricies will be saved. Default is "groups" dir under the input data directory.')
    parser.add_argument('--group-sizes', default='4,6,8', type=str, help='Sizes of groups, numbers divided by a comma. Ex.: 5,6,7,8')
    parser.add_argument('--num-groups-to-generate', default=1000, type=int, help='Number of groups to generate.')
    parser.add_argument('--num-of-candidates', default=1000, type=int, help='Number of candidates from all users that will be drawn randomly from which to select the best k ones.')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input), 'groups')

    args.group_sizes_list = list(map(lambda x: int(str.strip(x)), args.group_sizes.split(',')))

    return args


if __name__ == '__main__':
    args = parse_args()

    dataset = Dataset.load_dataset(path=args.input)
    for group_size in args.group_sizes_list:
        SampledTopKGroupGenerator().generate_groups(
            dataset=dataset,
            output_dir=args.output_dir,
            group_size=group_size,
            num_of_groups_to_generate=args.num_groups_to_generate,
        )
