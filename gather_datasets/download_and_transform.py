from os import path

from kgrec import KGRec
from movie_lens import MovieLens
from netflix import Netflix
from utils import create_directory



data_dir = '../.data'
destination_data_dir = '../datasets'

if __name__ == '__main__':
    create_directory(data_dir)

    print('─────────── Download and process Movie Lens dataset ───────────')
    MovieLens.download_dataset(data_dir)
    MovieLens.process_dataset(data_dir, destination_dir=path.join(destination_data_dir, 'movie_lens'))

    print('─────────── Download and process KGRec dataset ───────────')
    KGRec.download_dataset(data_dir)
    KGRec.process_dataset(data_dir, destination_dir=path.join(destination_data_dir, 'kgrec'))

    print('─────────── Download and process Netflix dataset ───────────')
    Netflix.download_dataset(data_dir)
    Netflix.process_dataset(data_dir, destination_dir=path.join(destination_data_dir, 'netflix'))
