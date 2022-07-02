from enum import Enum
import pandas as pd
from dataset import Dataset
from utils import IdMap
from utils import download_file_and_unzip, save_dataset

#define enum for movielenst sizes
class MovieLensSize(Enum):
    hundred_k = 'ml-100k'
    one_million = 'ml-1m'
    ten_million = 'ml-10m'
    twenty_million = 'ml-20m'
    twenty_five_million = 'ml-25m'

class MovieLens(Dataset):

    def __init__(self, size=MovieLensSize.hundred_k):
        self.size = size

    def download_dataset(self, data_dir):
        print('Downloading MovieLens dataset...')
        url = f'http://files.grouplens.org/datasets/movielens/{self.size.value}.zip'
        download_file_and_unzip(url, data_dir + '/movie_lens', f'{self.size.value}.zip')

    def process_dataset(self, data_dir, destination_dir, compress=True):
        print('Processing MovieLens dataset...')

        user_id_map = IdMap()
        item_id_map = IdMap()

        print('Transforming ratings.csv...')
        ratings_df = None
        if self.size == MovieLensSize.hundred_k or self.size == MovieLensSize.one_million:
            ratings_df = pd.read_csv(
                f'{data_dir}/movie_lens/{self.size.value}/ratings.dat',
                delimiter="::",
                engine='python',
                encoding='latin-1',
                header=None,
                names=['userId', 'movieId', 'rating', 'timestamp'])
        else:
            ratings_df = pd.read_csv(f'{data_dir}/movie_lens/{self.size.value}/ratings.csv')

        ratings_df.rename(columns={'movieId': 'item_id', 'userId': 'user_id'}, inplace=True)
        ratings_df['user_id'] = ratings_df['user_id'].map(user_id_map.map)
        ratings_df['item_id'] = ratings_df['item_id'].map(item_id_map.map)
        save_dataset(ratings_df, destination_dir, 'ratings', compress)

        print('Transforming movies.csv...')
        if self.size == MovieLensSize.hundred_k or self.size == MovieLensSize.one_million:
            movies_df = pd.read_csv(f'{data_dir}/movie_lens/{self.size.value}/movies.dat',
                delimiter="::",
                engine='python',
                encoding='latin-1',
                header=None,
                names=['movieId', 'title', 'genres'])
        else:
            movies_df = pd.read_csv(f'{data_dir}/movie_lens/{self.size.value}/movies.csv')

        genres_df = movies_df[['movieId', 'genres']]
        genres_df['genre'] = genres_df['genres'].str.split('|')
        genres_df = genres_df.explode('genre')[['movieId', 'genre']]
        genres_df.rename(columns={'movieId': 'item_id'}, inplace=True)
        genres_df['item_id'] = genres_df['item_id'].map(item_id_map.map)
        save_dataset(genres_df, destination_dir, 'genres', compress)

        movies_df.drop(columns=['genres'], inplace=True)
        movies_df.rename(columns={'movieId': 'item_id'}, inplace=True)
        movies_df['item_id'] = movies_df['item_id'].map(item_id_map.map)
        save_dataset(movies_df, destination_dir, 'movies', compress)

        # smaller datasets does not contain tags and links
        if self.size == MovieLensSize.hundred_k or self.size == MovieLensSize.one_million:
            return

        print('Transforming tags.csv...')
        tags_df = pd.read_csv(f'{data_dir}/movie_lens/{self.size.value}/tags.csv')
        tags_df.rename(columns={'movieId': 'item_id', 'userId': 'user_id'}, inplace=True)
        tags_df['user_id'] = tags_df['user_id'].map(user_id_map.map)
        tags_df['item_id'] = tags_df['item_id'].map(item_id_map.map)
        save_dataset(tags_df, destination_dir, 'tags', compress)

        print('Transforming links.csv...')
        links_df = pd.read_csv(f'{data_dir}/movie_lens/{self.size.value}/links.csv',
                               dtype={'movieId': 'int', 'imdbId': 'object', 'tmdbId': 'object'})
        links_df.rename(columns={'movieId': 'item_id', 'imdbId': 'imdb_id', 'tmdbId': 'tmdb_id'}, inplace=True)
        links_df['item_id'] = links_df['item_id'].map(item_id_map.map)
        save_dataset(links_df, destination_dir, 'links', compress)

        print('Transforming genome-scores.csv...')
        genome_scores_df = pd.read_csv(f'{data_dir}/movie_lens/{self.size.value}/genome-scores.csv')
        genome_scores_df.rename(columns={'movieId': 'item_id', 'tagId': 'tag_id'}, inplace=True)
        genome_scores_df['item_id'] = genome_scores_df['item_id'].map(item_id_map.map)
        save_dataset(genome_scores_df, destination_dir, 'genome_scores', compress)

        print('Transforming genome-tags.csv...')
        genome_tags_df = pd.read_csv(f'{data_dir}/movie_lens/{self.size.value}/genome-tags.csv')
        genome_tags_df.rename(columns={'tagId': 'tag_id'}, inplace=True)
        save_dataset(genome_tags_df, destination_dir, 'genome_tags', compress)
