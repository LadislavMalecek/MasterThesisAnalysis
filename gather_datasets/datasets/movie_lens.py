import pandas as pd
from dataset import Dataset
from utils import IdMap
from utils import download_file_and_unzip, save_dataset


class MovieLens(Dataset):
    def download_dataset(self, data_dir):
        print('Downloading MovieLens dataset...')
        url = 'http://files.grouplens.org/datasets/movielens/ml-25m.zip'
        download_file_and_unzip(url, data_dir + '/movie_lens', 'ml-25m.zip')

    def process_dataset(self, data_dir, destination_dir, compress=True):
        print('Processing MovieLens dataset...')

        user_id_map = IdMap()
        item_id_map = IdMap()


        print('Transforming ratings.csv...')
        ratings_df = pd.read_csv(f'{data_dir}/movie_lens/ml-25m/ratings.csv')
        ratings_df.rename(columns={'movieId': 'item_id', 'userId': 'user_id'}, inplace=True)
        ratings_df['user_id'] = ratings_df['user_id'].map(user_id_map.map)
        ratings_df['item_id'] = ratings_df['item_id'].map(item_id_map.map)
        save_dataset(ratings_df, destination_dir, 'ratings', compress)

        print('Transforming movies.csv...')
        movies_df = pd.read_csv(f'{data_dir}/movie_lens/ml-25m/movies.csv')
        genres_df = movies_df[['movieId', 'genres']]
        genres_df['genre'] = genres_df['genres'].str.split('|')
        genres_df = genres_df.explode('genre')
        genres_df.rename(columns={'movieId': 'item_id'}, inplace=True)
        genres_df['item_id'] = genres_df['item_id'].map(item_id_map.map)
        save_dataset(genres_df, destination_dir, 'genres', compress)

        movies_df.drop(columns=['genres'], inplace=True)
        movies_df.rename(columns={'movieId': 'item_id'}, inplace=True)
        movies_df['item_id'] = movies_df['item_id'].map(item_id_map.map)
        save_dataset(movies_df, destination_dir, 'movies', compress)

        print('Transforming tags.csv...')
        tags_df = pd.read_csv(f'{data_dir}/movie_lens/ml-25m/tags.csv')
        tags_df.rename(columns={'movieId': 'item_id', 'userId': 'user_id'}, inplace=True)
        tags_df['user_id'] = tags_df['user_id'].map(user_id_map.map)
        tags_df['item_id'] = tags_df['item_id'].map(item_id_map.map)
        save_dataset(tags_df, destination_dir, 'tags', compress)

        print('Transforming links.csv...')
        links_df = pd.read_csv(f'{data_dir}/movie_lens/ml-25m/links.csv',
                               dtype={'movieId': 'int', 'imdbId': 'object', 'tmdbId': 'object'})
        links_df.rename(columns={'movieId': 'item_id', 'imdbId': 'imdb_id', 'tmdbId': 'tmdb_id'}, inplace=True)
        links_df['item_id'] = links_df['item_id'].map(item_id_map.map)
        save_dataset(links_df, destination_dir, 'links', compress)

        print('Transforming genome-scores.csv...')
        genome_scores_df = pd.read_csv(f'{data_dir}/movie_lens/ml-25m/genome-scores.csv')
        genome_scores_df.rename(columns={'movieId': 'item_id', 'tagId': 'tag_id'}, inplace=True)
        genome_scores_df['item_id'] = genome_scores_df['item_id'].map(item_id_map.map)
        save_dataset(genome_scores_df, destination_dir, 'genome_scores', compress)

        print('Transforming genome-tags.csv...')
        genome_tags_df = pd.read_csv(f'{data_dir}/movie_lens/ml-25m/genome-tags.csv')
        genome_tags_df.rename(columns={'tagId': 'tag_id'}, inplace=True)
        save_dataset(genome_tags_df, destination_dir, 'genome_tags', compress)
