from os import path
import os

from kgrec import KGRec
from movie_lens import MovieLens
from netflix import Netflix
from utils import create_directory
import argparse


DEFAULT_PROCESSED_DATA_DIR = 'datasets'


COMPRESS_PARAMETER_DEFAULT = True
DESCRIPTION = 'Program to download and process common recommendation systems datasets.' \
    ' It downloads the datasets from the original sources and therefore does not violate the authors\' licences.' \
    ' After downloading, it transforms the datasets into a format that is easier to process and use for building recommender systems models.'

SUPPORTED_DATASETS = [ 'movie_lens', 'netflix', 'kgrec' ]

def parse_args():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--compress', default=COMPRESS_PARAMETER_DEFAULT, action=argparse.BooleanOptionalAction, help='Compress the datasets using gzip.')
    parser.add_argument('--data-dir', default=DEFAULT_PROCESSED_DATA_DIR, type=str, help=f'Path to the directory where the transformed datasets will be saved to, default: {DEFAULT_PROCESSED_DATA_DIR}')
    # parser for which datasets to download and process
    parser.add_argument('--datasets', default='all', type=str.lower, help=f'Comma separated list of datasets to download and process, default: all, supported: {SUPPORTED_DATASETS}')
    parser.add_argument('--download-only', default=False, action='store_true', help=f'Only download the datasets. Without processing.')
    args = parser.parse_args()

    datasets = args.datasets.split(sep=',')

    if 'all' in datasets:
        datasets = SUPPORTED_DATASETS

    for dataset in datasets:
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f'Dataset {dataset} not supported. Supported datasets: {SUPPORTED_DATASETS}')

    args.datasets = datasets

    # get absolute paths for raw and processed data directories
    args.data_dir = path.abspath(args.data_dir)

    return args


if __name__ == '__main__':
    args = parse_args()
    download_data_dir = path.join(args.data_dir, 'downloads')

    print(args)

    create_directory(args.data_dir)
    create_directory(download_data_dir)


    if 'movie_lens' in args.datasets:
        print('─────────── Download and process Movie Lens dataset ───────────')
        MovieLens.download_dataset(download_data_dir)
        if not args.download_only:
            MovieLens.process_dataset(
                data_dir=download_data_dir,
                destination_dir=path.join(args.data_dir, 'movie_lens'),
                compress=args.compress
            )
    

    if 'kgrec' in args.datasets:
        print('─────────── Download and process KGRec dataset ───────────')
        KGRec.download_dataset(download_data_dir)
        if not args.download_only:
            KGRec.process_dataset(
                data_dir=download_data_dir,
                destination_dir=path.join(args.data_dir, 'kgrec'),
                compress=args.compress
            )

    if 'netflix' in args.datasets:
        print('─────────── Download and process Netflix dataset ───────────')
        Netflix.download_dataset(download_data_dir)
        if not args.download_only:
            Netflix.process_dataset(
                data_dir=download_data_dir,
                destination_dir=path.join(args.data_dir, 'netflix'),
                compress=args.compress
            )
