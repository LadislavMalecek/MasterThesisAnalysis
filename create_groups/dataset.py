import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_array
from tqdm import tqdm


class Dataset:
    """
    This class represents a dataset with all required information needed to retrieve it and use it.
    Initialize it using the static method load_dataset(...).
    """

    def __init__(
            self,
            path: str,
    ) -> None:
        self.path = path
        self.dataset_df = None
        self.users_to_items_ratings_csr = None

    @staticmethod
    def load_dataset(path: str) -> 'Dataset':
        dataset = Dataset(path)
        print(f'Loading dataset...')
        dataset._load_dataset()
        print(f'Checking if ids are sequential...')
        dataset._check_if_ids_sequential()
        print(f'Creating sparse representation...')
        dataset._transform_to_sparse_matrix()
        return dataset

    def _load_dataset(self) -> None:
        self.dataset_df = pd.read_csv(self.path)

    def _check_if_ids_sequential(self) -> None:
        max_user_id = self.dataset_df['user_id'].max()
        max_item_id = self.dataset_df['item_id'].max()
        count_user_id = self.dataset_df['user_id'].nunique()
        count_item_id = self.dataset_df['item_id'].nunique()

        print(f'Max user id: {max_user_id}')
        print(f'Max item id: {max_item_id}')
        print(f'Count user id: {count_user_id}')
        print(f'Count item id: {count_item_id}')

        assert max_user_id == (count_user_id - 1)
        assert max_item_id == (count_item_id - 1)

    def _transform_to_sparse_matrix(self) -> csr_matrix:
        max_user_id = self.dataset_df['user_id'].max()
        max_item_id = self.dataset_df['item_id'].max()

        has_rating = 'rating' in self.dataset_df.columns

        users_to_items_ratings = lil_array((max_user_id + 1, max_item_id + 1), dtype=np.float32)
        if has_rating:
            for user_id, item_id, rating, *_ in tqdm(self.dataset_df.itertuples(index=False), total=self.dataset_df.shape[0]):
                users_to_items_ratings[user_id, item_id] = rating
        else:
            for user_id, item_id, *_ in tqdm(self.dataset_df.itertuples(index=False), total=self.dataset_df.shape[0]):
                users_to_items_ratings[user_id, item_id] = 1

        self.users_to_items_ratings_csr = users_to_items_ratings.tocsr()
        # save_npz(f'npz/sparse_datasets/{self.name}_users_to_items_ratings.npz', self.users_to_items_ratings_csr)
        return self.users_to_items_ratings_csr

    def get_number_of_users(self) -> int:
        return self.users_to_items_ratings_csr.shape[0]

    def get_number_of_items(self) -> int:
        return self.users_to_items_ratings_csr.shape[1]
