from os import path

import pandas as pd
from dataset import Dataset
from utils import download_file_and_unzip, get_files, save_dataset


class KGRec(Dataset):
    def download_dataset(self, data_dir):
        print('Downloading GFar dataset...')
        url = 'http://mtg.upf.edu/system/files/projectsweb/KGRec-dataset.zip'
        download_file_and_unzip(url, data_dir + '/kgrec', 'KGRec-dataset.zip')

    def process_dataset(self, data_dir, destination_dir, compress=True):
        for dataset in ['music', 'sound']:
            dataset_dir = path.join(
                data_dir, 'kgrec', 'KGRec-dataset', f'KGRec-{dataset}')

            # Load ratings, remove the rating column which is always 1
            ratings_file = 'implicit_lf_dataset.csv' if dataset == 'music' else 'downloads_fs_dataset.txt'
            print(f'Processing KGRec-{dataset} dataset...')
            ratings_df = pd.read_csv(path.join(dataset_dir, ratings_file), sep='\t', header=None, names=['user_id', 'item_id', 'rating'])
            ratings_df.drop(columns=['rating'], inplace=True)
            ratings_df.sort_values(by=['user_id', 'item_id'], inplace=True)
            save_dataset(ratings_df, destination_dir, f'{dataset}_ratings', compress)

            # Load descriptions and tags files, relpace nonstandard quotes with standard ones and join them together
            for other_data_file, column_name in [('descriptions', 'description'), ('tags', 'tag')]:
                data = []
                dir = str(path.join(dataset_dir, other_data_file))
                files = get_files(dir)
                for file in files:
                    id = int(file.split(sep='/')[-1].split('.')[0])
                    with open(file) as f:
                        lines = f.readlines()
                        cleaned_lines = [line.strip().replace('`', "'").replace("''", "'") for line in lines]
                        cleaned_text = ' '.join(cleaned_lines)
                        data.append((id, cleaned_text))

                df = pd.DataFrame(data=data, columns=['item_id', column_name])
                if column_name == 'tag':
                    df['tag_n'] = df['tag'].str.split(' ')
                    df = df.explode('tag_n')
                    df.drop(columns=['tag'], inplace=True)
                    df.rename(columns={'tag_n': 'tag'}, inplace=True)
                df.sort_values('item_id', inplace=True)
                df.reset_index(drop=True)
                save_dataset(df, destination_dir,
                             f'{dataset}_{column_name}s', compress)
