import argparse
import os

import pandas as pd

from create_groups.group_generator import GroupGenerator
from create_groups.dataset import Dataset
from create_groups.similarity_sampler_tools import SimilaritySamplerTools

DESCRIPTION = 'Generates groups for a dataset based on random selection.'


class RandomGroupGenerator(GroupGenerator):
    """
    Simplest group generator. It generates tuples of random users.
    """

    def generate_groups(
            self,
            dataset: Dataset,
            output_dir: str,
            group_size: int,
            num_of_groups_to_generate: int,
    ) -> pd.DataFrame:
        print(f"Generating random groups")

        sim_sampler_tools = SimilaritySamplerTools(dataset)
        groups = [sim_sampler_tools.draw_unique_candidates(group_size) for _ in range(num_of_groups_to_generate)]

        df = pd.DataFrame(groups)
        self.save_group_df(df, output_dir=output_dir, group_type='random', group_size=group_size)
        return df


def parse_args():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--input', default='../datasets/kgrec/music_ratings.csv.gz', help='The dataset to use, needs to be csv dataframe with columns "user_id", "item_id" and optionally "rating".')
    parser.add_argument('--output-dir', default=None, help='The directory where the resulting matricies will be saved. Default is "groups" dir under the input data directory.')
    parser.add_argument('--group-sizes', default='4,6,8', type=str, help='Sizes of groups, numbers divided by a comma. Ex.: 5,6,7,8')
    parser.add_argument('--num-groups-to-generate', default=1000, type=int, help='Number of groups to generate.')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input), 'groups')

    args.group_sizes_list = list(map(lambda x: int(str.strip(x)), args.group_sizes.split(',')))

    return args


if __name__ == '__main__':
    args = parse_args()

    dataset = Dataset.load_dataset(path=args.input)
    for group_size in args.group_sizes_list:
        RandomGroupGenerator().generate_groups(
            dataset=dataset,
            output_dir=args.output_dir,
            group_size=group_size,
            num_of_groups_to_generate=args.num_groups_to_generate,
        )
