from os import path

from datasets.lastfm import LastFM
from datasets.kgrec import KGRec
from datasets.movie_lens import MovieLens
from datasets.netflix import Netflix
from datasets.spotify import Spotify

from utils import create_directory
import argparse


DEFAULT_PROCESSED_DATA_DIR = 'datasets'


COMPRESS_PARAMETER_DEFAULT = True
DESCRIPTION = 'Program to download and process common recommendation systems datasets.' \
    ' It downloads the datasets from the original sources and therefore does not violate the authors\' licences.' \
    ' After downloading, it transforms the datasets into a format that is easier to process and use for building recommender systems models.'

DATASET_PROCESSORS = {
    'movie_lens': MovieLens,
    'netflix': Netflix,
    'lastfm': LastFM,
    'kgrec': KGRec,
    'spotify': Spotify,
}


def parse_args():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--compress', default=COMPRESS_PARAMETER_DEFAULT, action=argparse.BooleanOptionalAction, help='Compress the datasets using gzip.')
    parser.add_argument('--data-dir', default=DEFAULT_PROCESSED_DATA_DIR, type=str, help=f'Path to the directory where the transformed datasets will be saved to, default: {DEFAULT_PROCESSED_DATA_DIR}')
    # parser for which datasets to download and process
    parser.add_argument('--datasets', default='all', type=str.lower, help=f'Comma separated list of datasets to download and process, default: all, supported: {DATASET_PROCESSORS.keys}')
    parser.add_argument('--download-only', default=False, action='store_true', help=f'Only download the datasets. Without processing.')
    args = parser.parse_args()

    datasets = args.datasets.split(sep=',')

    # check if 'all' present, if so, replace with all available datasets
    if 'all' in datasets:
        datasets = DATASET_PROCESSORS.keys()

    # check if all datasets are supported
    for dataset in datasets:
        if dataset not in DATASET_PROCESSORS.keys():
            raise ValueError(
                f'Dataset {dataset} not supported. Supported datasets: {DATASET_PROCESSORS.keys()}')

    # replace args.datasets with ordered unique list of datasets
    args.datasets = list(sorted(set(datasets)))

    # get absolute paths for raw and processed data directories
    args.data_dir = path.abspath(args.data_dir)
    return args


if __name__ == '__main__':
    args = parse_args()
    download_data_dir = path.join(args.data_dir, 'downloads')

    create_directory(args.data_dir)
    create_directory(download_data_dir)

    for dataset in args.datasets:
        print(f'┌─────────── ⚙️  Downloading {dataset} dataset ───────────')
        dataset_processor = DATASET_PROCESSORS[dataset]()
        dataset_processor.download_dataset(download_data_dir)
        if args.download_only:
            exit()

        print(f'├─────────── ⚙️  Transforming {dataset} dataset')
        dataset_processor.process_dataset(
            data_dir=download_data_dir,
            destination_dir=path.join(args.data_dir, dataset),
            compress=args.compress
        )
        print(f'└─────────── ✅ Finished processing {dataset} dataset ───────────')
