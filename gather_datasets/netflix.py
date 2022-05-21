from os import path

import pandas as pd
from tqdm import tqdm

from utils import create_directory, download_file_with_progress, extract_file_with_progress, get_files, save_dataset


class Netflix:
    @staticmethod
    def download_dataset(data_dir):
        print('Downloading Netflix dataset...')
        url = 'https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz'
        file_name = 'nf_prize_dataset.tar.gz'
        netflix_data_dir = path.join(data_dir, 'netflix')
        file_path = path.join(netflix_data_dir, file_name)
        inner_tar_path = path.join(netflix_data_dir, 'download', 'training_set.tar')

        create_directory(netflix_data_dir)
        download_file_with_progress(url, file_path)
        extract_file_with_progress(file_path, netflix_data_dir)
        extract_file_with_progress(inner_tar_path, netflix_data_dir)

    @staticmethod
    def process_dataset(data_dir, destination_dir, compress=True):
        print('Processing Netflix dataset...')
        netflix_data_dir = path.join(data_dir, 'netflix')
        ratings_files = get_files(path.join(netflix_data_dir, 'training_set'))
        ratings_files.sort()
        all_dataframes = []
        pbar = tqdm(total=len(ratings_files), unit='files', unit_scale=True, unit_divisor=1000)
        for ratings_file in ratings_files:
            file_name = path.basename(ratings_file)
            item_id = int(file_name[3:-4])
            # print(f'Processing {file_name}...')
            # skip first row, it is the id of the movie, which is the same as the file name
            ratings_df = pd.read_csv(
                ratings_file,
                header=None,
                skiprows=[0],
                names=['user_id', 'rating', 'date'],
                dtype={'user_id': 'int32', 'rating': 'int8', 'date': 'object'}
            )
            # add the id of the movie as a column
            ratings_df['item_id'] = item_id
            ratings_df = ratings_df[['user_id', 'item_id', 'rating', 'date']]
            all_dataframes.append(ratings_df)
            pbar.update(1)
        pbar.close()

        print('Concatenating all dataframes...')
        ratings_df = pd.concat(all_dataframes)
        ratings_df.sort_values(by=['user_id', 'item_id'], inplace=True)
        ratings_df.reset_index(drop=True)
        save_dataset(ratings_df, destination_dir, 'ratings', can_take_long=True)

        print('Transforming movies_titles.txt...')
        movies_titles_df = pd.read_csv(
            path.join(netflix_data_dir, 'download', 'movie_titles.txt'),
            header=None,
            names=['item_id', 'release_year', 'title'],
            dtype={'item_id': 'int16', 'release_year': 'Int16', 'title': 'object'},
            # utf-8 leads to 'cannot decode 0xe9' error
            encoding="ISO-8859-1",
            engine='python',
            # on bad line concatenate the tail rows together
            on_bad_lines=lambda x: [x[0], x[1], ','.join(x[2:])]
        )

        movies_titles_df.sort_values(by=['item_id'], inplace=True)
        movies_titles_df.reset_index(drop=True)
        save_dataset(movies_titles_df, destination_dir, 'movies', compress)
