from os import path

from kgrec import KGRec
from movie_lens import MovieLens
from netflix import Netflix
from utils import create_directory

# def create_directory(directory_path):
#     # create dir if not exist
#     dir_exists = path.exists(directory_path)
#     if not dir_exists:
#         print(f'Creating directory {path.abspath(directory_path)}...')
#         os.makedirs(directory_path, exist_ok=True)

# def download_file_with_progress(url, file_path):
#     # check if file already exists
#     # else download file
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         content_size = int(r.headers['content-length'])
#         if path.isfile(file_path) and path.getsize(file_path) == content_size:
#             print('File already exists. Skipping download.')
#             return
#         with open(file_path, 'wb') as f:
#             pbar = tqdm(total=content_size, unit='B', unit_scale=True, unit_divisor=1024)
#             for chunk in r.iter_content(chunk_size=1024 * 64):
#                 if chunk:  # filter out keep-alive new chunks
#                     f.write(chunk)
#                     pbar.update(len(chunk))

# def extract_file_with_progress(file_path, directory_path):
#     print(f'Extracting file {file_path}...')
#     if file_path.endswith('.zip'):
#         # Open your .zip file
#         with ZipFile(file=file_path) as zip_file:
#             # Loop over each file
#             pbar = tqdm(total=len(zip_file.namelist()), unit='files', unit_scale=True, unit_divisor=1000)
#             for file in zip_file.namelist():
#                 # Extract each file to another directory
#                 # If you want to extract to current working directory, don't specify path
#                 zip_file.extract(member=file, path=directory_path)
#                 pbar.update(1)

#     elif file_path.endswith('.tar.gz') or file_path.endswith('.tar'):
#         open_mode = 'r:gz' if file_path.endswith('.tar.gz') else 'r'
#         with tarfile.open(file_path, open_mode) as tar_file:
#             # Loop over each file
#             pbar = tqdm(total=len(zip_file.namelist()), unit='files', unit_scale=True, unit_divisor=1000)
#             for file in tar_file.getnames():
#                 # Extract each file to another directory
#                 tar_file.extract(member=file, path=directory_path)
#                 pbar.update(1)
#     else:
#         raise ValueError('File is not a .zip or .tar.gz file.')

# def get_files(dir):
#     files = [ dir + '/' + f for f in os.listdir(dir) if path.isfile(path.join(dir, f))]
#     return files

# def download_file_and_unzip(url, data_dir, file_name):
#     file_path = path.join(data_dir, file_name)
#     create_directory(data_dir)
#     download_file_with_progress(url, file_path)
#     extract_file_with_progress(file_path, data_dir)

# def save_dataset(dataset_to_save, destination_dir, dataset_name, can_take_long=False):
#     create_directory(destination_dir)
#     if can_take_long:
#         print(f'Saving {dataset_name} dataset, can take a long time...')
#     else:
#         print(f'Saving {dataset_name} dataset...')
#     # dataset_to_save.to_csv(path.join(destination_dir, dataset_name + '.csv.gz'), index=False, header=True, compression='gzip')
#     dataset_to_save.to_csv(path.join(destination_dir, dataset_name + '.csv'), index=False, header=True)

# def download_movie_lens():
#     print('Downloading MovieLens dataset...')
#     url = 'http://files.grouplens.org/datasets/movielens/ml-25m.zip'
#     download_file_and_unzip(url, data_dir + '/movie_lens', 'ml-25m.zip')

# def process_movie_lens_dataset(destination_dir):
#     print('Processing MovieLens dataset...')

#     print('Transforming ratings.csv...')
#     ratings_df = pd.read_csv('.data/movie_lens/ml-25m/ratings.csv')
#     ratings_df.rename(columns={'movieId': 'item_id', 'userId': 'user_id'}, inplace=True)
#     save_dataset(ratings_df, destination_dir, 'ratings')

#     print('Transforming movies.csv...')
#     movies_df = pd.read_csv('.data/movie_lens/ml-25m/movies.csv')
#     genres_df = movies_df[['movieId', 'genres']]
#     genres_df['genres'] = genres_df['genres'].str.split('|')
#     genres_df = genres_df.explode('genres')
#     genres_df.rename(columns={'movieId': 'item_id', 'genres': 'genre'}, inplace=True)
#     movies_df.drop(columns=['genres'], inplace=True)
#     movies_df.rename(columns={'movieId': 'item_id'}, inplace=True)
#     save_dataset(genres_df, destination_dir, 'genres')
#     save_dataset(movies_df, destination_dir, 'movies')

#     print('Transforming tags.csv...')
#     tags_df = pd.read_csv('.data/movie_lens/ml-25m/tags.csv')
#     tags_df.rename(columns={'movieId': 'item_id', 'userId': 'user_id'}, inplace=True)
#     save_dataset(tags_df, destination_dir, 'tags')

#     print('Transforming links.csv...')
#     links_df = pd.read_csv('.data/movie_lens/ml-25m/links.csv', dtype={'movieId': 'int', 'imdbId': 'object', 'tmdbId': 'object'})
#     links_df.rename(columns={'movieId': 'item_id', 'imdbId': 'imdb_id', 'tmdbId': 'tmdb_id'}, inplace=True)
#     save_dataset(links_df, destination_dir, 'links')

#     print('Transforming genome-scores.csv...')
#     genome_scores_df = pd.read_csv('.data/movie_lens/ml-25m/genome-scores.csv')
#     genome_scores_df.rename(columns={'movieId': 'item_id', 'tagId': 'tag_id'}, inplace=True)
#     save_dataset(genome_scores_df, destination_dir, 'genome_scores')

#     print('Transforming genome-tags.csv...')
#     genome_tags_df = pd.read_csv('.data/movie_lens/ml-25m/genome-tags.csv')
#     genome_tags_df.rename(columns={'tagId': 'tag_id'}, inplace=True)
#     save_dataset(genome_tags_df, destination_dir, 'genome_tags')


# def download_gfar_dataset():
#     print('Downloading GFar dataset...')
#     url = 'http://mtg.upf.edu/system/files/projectsweb/KGRec-dataset.zip'
#     download_file_and_unzip(url, data_dir + '/kgrec', 'KGRec-dataset.zip')


# def process_gfar_dataset(destination_dir):
#     for dataset in ['music', 'sound']:
#         dataset_dir = path.join(data_dir, 'kgrec', 'KGRec-dataset', f'KGRec-{dataset}')
#         ratings_file = 'implicit_lf_dataset.csv' if dataset == 'music' else 'downloads_fs_dataset.txt'
#         print(f'Processing KGRec-{dataset} dataset...')
#         ratings_df = pd.read_csv(path.join(dataset_dir, ratings_file), sep='\t', header=None, names=['user_id', 'item_id', 'rating'])
#         ratings_df.drop(columns=['rating'], inplace=True)
#         ratings_df.sort_values(by=['user_id', 'item_id'], inplace=True)
#         save_dataset(ratings_df, destination_dir, f'{dataset}_ratings')

#         for other_data_file, column_name in [('descriptions', 'description'), ('tags', 'tag')]:
#             data = []
#             dir = str(path.join(dataset_dir, other_data_file))
#             files = get_files(dir)
#             for file in files:
#                 id = int(file.split(sep='/')[-1].split('.')[0])
#                 with open(file) as f:
#                     lines = f.readlines()
#                     cleaned_lines = [line.strip().replace('`', "'").replace("''", "'") for line in lines]
#                     cleaned_text = ' '.join(cleaned_lines)
#                     data.append((id, cleaned_text))
#             df = pd.DataFrame(data=data, columns=['item_id', column_name])
#             df.sort_values('item_id', inplace=True)
#             df.reset_index(drop=True)
#             save_dataset(df, destination_dir, f'{dataset}_{column_name}')


# def download_netflix_dataset():
#     print('Downloading Netflix dataset...')
#     url = 'https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz'
#     file_name = 'nf_prize_dataset.tar.gz'
#     netflix_data_dir = path.join(data_dir, 'netflix')
#     file_path = path.join(netflix_data_dir, file_name)
#     inner_tar_path = path.join(netflix_data_dir, 'download', 'training_set.tar')

#     create_directory(netflix_data_dir)
#     download_file_with_progress(url, file_path)
#     extract_file_with_progress(file_path, netflix_data_dir)
#     extract_file_with_progress(inner_tar_path, netflix_data_dir)

# def process_netflix_dataset(destination_dir):
#     print('Processing Netflix dataset...')
#     netflix_data_dir = path.join(data_dir, 'netflix')
#     ratings_files = get_files(path.join(netflix_data_dir, 'training_set'))
#     ratings_files.sort()
#     all_dataframes = []
#     pbar = tqdm(total=len(ratings_files), unit='files', unit_scale=True, unit_divisor=1000)
#     for ratings_file in ratings_files:
#         file_name = path.basename(ratings_file)
#         id = int(file_name[3:-4])
#         # print(f'Processing {file_name}...')
#         # skip first row, it is the id of the movie, which is the same as the file name
#         ratings_df = pd.read_csv(
#             ratings_file,
#             header=None,
#             skiprows=[0],
#             names=['user_id', 'rating', 'date'],
#             dtype={'user_id': 'int32', 'rating': 'int8', 'date': 'object'}
#         )
#         # add the id of the movie as a column
#         ratings_df['item_id'] = id
#         ratings_df = ratings_df[['user_id', 'item_id', 'rating', 'date']]
#         all_dataframes.append(ratings_df)
#         pbar.update(1)
#     pbar.close()

# print('Concatenating all dataframes...')
# ratings_df = pd.concat(all_dataframes)
# ratings_df.sort_values(by=['user_id', 'item_id'], inplace=True)
# ratings_df.reset_index(drop=True)
# save_dataset(ratings_df, destination_dir, 'ratings', can_take_long=True)

# print('Transforming movies_titles.txt...')
# movies_titles_df = pd.read_csv(
#     path.join(netflix_data_dir, 'download', 'movie_titles.txt'),
#     header=None,
#     names=['item_id', 'release_year', 'title'],
#     dtype={'item_id': 'int16', 'release_year': 'Int16', 'title': 'object'},
#     # utf-8 leads to 'cannot decode 0xe9' error
#     encoding = "ISO-8859-1",
#     engine='python',
#     # on bad line concatenate the tail rows together
#     on_bad_lines=lambda x: [x[0], x[1], ','.join(x[2:])]
# )

# movies_titles_df.sort_values(by=['item_id'], inplace=True)
# movies_titles_df.reset_index(drop=True)
# save_dataset(movies_titles_df, destination_dir, 'movies')


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
