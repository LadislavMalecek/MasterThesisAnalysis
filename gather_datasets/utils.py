from collections import defaultdict
import collections
from os import path
import os
import tarfile
from zipfile import ZipFile
import requests
import hashlib

from tqdm import tqdm


def create_directory(directory_path):
    # create dir if not exist
    dir_exists = path.exists(directory_path)
    if not dir_exists:
        print(f'Creating directory {path.abspath(directory_path)}...')
        os.makedirs(directory_path, exist_ok=True)


def download_file_with_progress(url, file_path):
    # check if file already exists
    # else download file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        content_size = int(r.headers['content-length'])
        if path.isfile(file_path) and path.getsize(file_path) == content_size:
            print('File already exists. Skipping download.')
            return
        with open(file_path, 'wb') as f:
            pbar = tqdm(total=content_size, unit='B', unit_scale=True, unit_divisor=1024)
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))


def get_md5_hash(file_path):
    print('Calculating md5 hash...')
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_if_file_exists(file_path, md5_hash=None):
    if not path.isfile(file_path):
        return False
    if md5_hash is None:
        return True
    return md5_hash == get_md5_hash(file_path)

def check_if_file_valid(file_path, md5_hash):
    if not path.isfile(file_path):
        return (False, f'File {file_path} does not exist')
    hash_matches = md5_hash == get_md5_hash(file_path)
    if not hash_matches:
        return (False, f'File invalid, {file_path} hash does not match')
    return (True, 'File is valid')


def extract_file_with_progress(file_path, directory_path):
    print(f'Extracting file {file_path}...')
    if file_path.endswith('.zip'):
        # Open your .zip file
        with ZipFile(file=file_path) as zip_file:
            # Loop over each file
            pbar = tqdm(total=len(zip_file.namelist()), unit='files', unit_scale=True, unit_divisor=1000)
            for file in zip_file.namelist():
                # Extract each file to another directory
                # If you want to extract to current working directory, don't specify path
                zip_file.extract(member=file, path=directory_path)
                pbar.update(1)
            pbar.close()

    elif file_path.endswith('.tar.gz') or file_path.endswith('.tar'):
        open_mode = 'r:gz' if file_path.endswith('.tar.gz') else 'r'
        with tarfile.open(file_path, open_mode) as tar_file:
            # Loop over each file
            pbar = tqdm(total=len(tar_file.getnames()), unit='files', unit_scale=True, unit_divisor=1000)
            for file in tar_file.getnames():
                # Extract each file to another directory
                tar_file.extract(member=file, path=directory_path)
                pbar.update(1)
            pbar.close()
    else:
        raise ValueError('File is not a .zip or .tar.gz file.')


def get_files(dir):
    files = [dir + '/' + f for f in os.listdir(dir) if path.isfile(path.join(dir, f))]
    return files


def download_file_and_unzip(url, data_dir, file_name):
    file_path = path.join(data_dir, file_name)
    create_directory(data_dir)
    download_file_with_progress(url, file_path)
    extract_file_with_progress(file_path, data_dir)


def save_dataset(dataset_to_save, destination_dir, dataset_name, compress=True, can_take_long=False):
    create_directory(destination_dir)

    destination_file = path.join(destination_dir, dataset_name + '.csv')
    if compress:
        destination_file += '.gz'

    if can_take_long:
        print(f'Saving {dataset_name} dataset to {destination_file}, can take a long time...')
    else:
        print(f'Saving {dataset_name} dataset to {destination_file}...')

    if compress:
        dataset_to_save.to_csv(destination_file, index=False, header=True, compression='gzip')
    else:
        dataset_to_save.to_csv(destination_file, index=False, header=True)


class IdMap(dict):
    def __init__(self, *args, **kwargs):
        self._original_to_int_id_map = {}
        self._last_id = -1
        
    def _get_next_id_value(self):
        self._last_id += 1
        return self._last_id

    def map(self, key):
        if key not in self._original_to_int_id_map:
            self._original_to_int_id_map[key] = self._get_next_id_value()
        return self._original_to_int_id_map[key]
