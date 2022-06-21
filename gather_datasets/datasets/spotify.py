from collections import namedtuple
import json
from os import path

import pandas as pd
from tqdm import tqdm
from dataset import Dataset
from utils import extract_file_with_progress
from utils import check_if_file_valid, get_files, save_dataset

class Spotify(Dataset):

    DATASET_MD5 = 'a7f47d6744195e421ee8d43a820963b1'

    def download_dataset(self, data_dir):
        file_path = path.join(data_dir, 'spotify', 'spotify_million_playlist_dataset.zip')

        file_ok = False
        while True:
            file_ok, error = check_if_file_valid(file_path, Spotify.DATASET_MD5)
            if file_ok:
                # while loop exit condition
                break
            # file not ok, print out instructions and wait for user input
            print(error)
            print('Unfortunatelly Spotify dataset is not available for download without an account at https://www.aicrowd.com/')
            print(f'Please, download the dataset manually from https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files and place it in the following directory: {file_path}')
            print('After downloading, and placing the dataset in the directory, press Enter to continue.')
            user_input = input('Enter to continue, \'exit\' to stop processing of Spotify dataset.')
            if user_input == 'exit':
                return False

        extract_file_with_progress(file_path, data_dir + '/spotify')

    def process_dataset(self, data_dir, destination_dir, compress=True):
        # dataset is comprised of lots of json files that contain each many playlists
        # we need to merge them into a single csv file
        print('Processing Spotify dataset...')
        TrackInfo = namedtuple('TrackInfo', ['item_id', 'item_uri', 'item_name', 'artist_uri', 'artist_name', 'album_uri'])

        tracks = {}
        playlists = []
        next_id = 0

        netflix_data_dir = path.join(data_dir, 'spotify')
        ratings_files = get_files(path.join(netflix_data_dir, 'data'))
        pbar = tqdm(total=len(ratings_files), unit='files', unit_scale=True, unit_divisor=1000)
        for ratings_file in ratings_files:
            data = None
            with open(ratings_file, 'r') as f:
                data = json.load(f)
                playlists_data = data['playlists']
                for playlist in playlists_data:
                    # timestamp = playlist['modified_at']
                    pid = playlist['pid']
                    tracks_data = playlist['tracks']
                    for track in tracks_data:
                        track_uri = track['track_uri']
                        
                        track_id = None
                        if track_uri in tracks:
                            track_id = tracks[track_uri].item_id
                        else:
                            track_id = next_id
                            next_id += 1
                            tracks[track_uri] = TrackInfo(track_id, track_uri, track['track_name'], track['artist_uri'], track['artist_name'], track['album_uri'])
                        playlists.append((pid, track_id))
            pbar.update(1)
        pbar.close()

        print('Transforming Spotify dataset...')
        tracks_df = pd.DataFrame.from_records(list(tracks.values()), columns=TrackInfo._fields)
        tracks_df.reset_index(inplace=True, drop=True)
        save_dataset(tracks_df, destination_dir, 'tracks', compress)
        ratings_df = pd.DataFrame.from_records(playlists, columns=['playlist_id', 'item_id'])
        save_dataset(ratings_df, destination_dir, 'ratings', compress)
