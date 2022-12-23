import argparse
import os
import numpy as np


def generate(group_size, num_groups, file_name):
    rand = np.random.rand(num_groups, group_size)
    array = np.divide(rand, rand.sum(axis=1).reshape((-1, 1)))
    np.savetxt(file_name, array, delimiter=',', fmt='%.2f')


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--output-dir', help='The directory where the resulting matrices will be saved. Default is "groups" dir under the input data directory.')
        parser.add_argument('--group-sizes', default='4,6,8', type=str, help='Sizes of groups, numbers divided by a comma. Ex.: 5,6,7,8')
        parser.add_argument('--num-groups-to-generate', default=1000, type=int, help='Number of groups to generate.')

        args = parser.parse_args()
        args.group_sizes_list = list(map(lambda x: int(str.strip(x)), args.group_sizes.split(',')))
        return args

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for g_size in args.group_sizes_list:
        filename = f'group_weights_{str(g_size)}.csv'
        file = os.path.join(args.output_dir, filename)
        generate(g_size, 1000, file)
