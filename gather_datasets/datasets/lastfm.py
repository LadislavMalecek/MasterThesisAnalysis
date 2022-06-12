from collections import defaultdict
from os import path
import pandas as pd
from dataset import Dataset
from utils import IdMap
from utils import download_file_and_unzip, save_dataset


class LastFM(Dataset):
    def download_dataset(self, data_dir):
        print('Downloading LastFM dataset...')
        url = 'http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz'
        download_file_and_unzip(url, data_dir + '/lastfm', 'lastfm-dataset-360K.tar.gz')

    def process_dataset(self, data_dir, destination_dir, compress=True):
        print('Processing LastFM dataset...')

        user_id_map = IdMap()
        item_id_map = IdMap()

        print('Transforming ratings.csv...')
        plays_ratings_df = pd.read_csv(
            path.join(data_dir, 'lastfm', 'lastfm-dataset-360K', 'usersha1-artmbid-artname-plays.tsv'),
            delimiter='\t', 
            header=None, 
            names=['user_id', 'item_id', 'artist_name', 'plays']
        )

        # create new id for each user
        plays_ratings_df['user_id'] = plays_ratings_df['user_id'].map(user_id_map.map)
        plays_ratings_df['item_id'] = plays_ratings_df['item_id'].map(item_id_map.map)

        ratings_df = plays_ratings_df[['user_id', 'item_id', 'plays']]
        save_dataset(ratings_df, destination_dir, 'ratings', compress)

        items_df = plays_ratings_df[['item_id', 'artist_name']].drop_duplicates()
        save_dataset(items_df, destination_dir, 'items', compress)

        users_df = pd.read_csv(
            path.join(data_dir, 'lastfm', 'lastfm-dataset-360K', 'usersha1-profile.tsv'),
            delimiter='\t', 
            header=None, 
            names=['user_id', 'gender', 'age', 'country', 'signup']
        )

        # create new id for each user using the same user_id_map
        users_df['user_id'] = users_df['user_id'].map(user_id_map.map)

        save_dataset(users_df, destination_dir, 'users', compress)

