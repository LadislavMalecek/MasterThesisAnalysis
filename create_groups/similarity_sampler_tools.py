import os
import random
from typing import List, Optional

import numpy as np
import pandas as pd
from more_itertools import chunked
from ray.util.client import ray
from tqdm import tqdm

from create_groups.dataset import Dataset


class SimilaritySamplerTools:
    """
    Colection of tools used to sample and calculate similarities between users for a specific dataset.
    """

    def __init__(self, dataset: Dataset, ray_threads: Optional[int] = None) -> None:
        self.dataset = dataset
        self._dataset_df = dataset.dataset_df
        self._users_to_items_ratings_csr = dataset.users_to_items_ratings_csr
        self._num_users = dataset.get_number_of_users()
        self.ray_threads = ray_threads

    def init_ray(self):
        if not ray.is_initialized():
            ray.init(num_cpus=self.ray_threads)

    def get_random_users(self):
        while True:
            u1, u2 = random.randint(0, self._num_users - 1), random.randint(0, self._num_users - 1)
            if u1 != u2:
                return u1, u2

    @staticmethod
    @ray.remote
    def _calculate_similarity_ray(u1_ratings, u2_ratings) -> float:
        return SimilaritySamplerTools._calculate_similarity(u1_ratings, u2_ratings)

    @staticmethod
    def _calculate_similarity(u1_ratings, u2_ratings) -> float:
        u1_n = u1_ratings.dot(u1_ratings.T)[0, 0]
        if u1_n == 0:
            return 0
        u2_n = u2_ratings.dot(u2_ratings.T)[0, 0]
        if u2_n == 0:
            return 0
        denominator = np.sqrt(u1_n) * np.sqrt(u2_n)
        nominator = u1_ratings.dot(u2_ratings.T)[0, 0]
        result = nominator / denominator
        return result

    def draw_random_samples(self, num_samples: int) -> List[float]:
        self.init_ray()
        samples = []
        random_tuples = list(map(lambda _: self.get_random_users(), range(num_samples)))
        # take n random samples
        with tqdm(total=num_samples) as progress_bar:
            for chunk in chunked(random_tuples, 1000):
                ray_results = []
                for u1, u2 in chunk:
                    u1_ratings = self._users_to_items_ratings_csr.getrow(u1)
                    u2_ratings = self._users_to_items_ratings_csr.getrow(u2)
                    ray_results.append(self._calculate_similarity_ray.remote(u1_ratings, u2_ratings))
                results = ray.get(ray_results)
                samples.extend(results)
                progress_bar.update(len(results))
        return samples

    def draw_unique_candidates(self, num_of_candidates: int) -> List[int]:
        return random.sample(range(self._num_users), num_of_candidates)

    def calculate_similarity_of_candidates_parallel(self, pivot_id, candidates_ids):
        self.init_ray()
        ray_results = []
        ray_parameters = []

        pivot_ratings = self.dataset.users_to_items_ratings_csr.getrow(pivot_id)
        for candidate_id in candidates_ids:
            candidate_ratings = self._users_to_items_ratings_csr.getrow(candidate_id)
            ray_results.append(self._calculate_similarity_ray.remote(pivot_ratings, candidate_ratings))
            ray_parameters.append(candidate_id)

        results = ray.get(ray_results)
        return zip(ray_parameters, results)

    def calculate_similarity_of_candidates(self, pivot_id, candidates_ids):
        results = []
        pivot_ratings = self.dataset.users_to_items_ratings_csr.getrow(pivot_id)
        for candidate_id in candidates_ids:
            candidate_ratings = self._users_to_items_ratings_csr.getrow(candidate_id)
            sim = self._calculate_similarity(pivot_ratings, candidate_ratings)
            results.append((candidate_id, sim))
        return results

    def get_similarity_sampling(
            self,
            num_of_samples,
            cache_dir,
            filter_out_zero_similarity=True,
    ) -> List[float]:
        file = os.path.join(cache_dir, f'{num_of_samples}.csv')
        if os.path.exists(file):
            print(f'Loading cached similarity sampling from file: {file}')
            samples = pd.read_csv(file)['similarity'].tolist()
            if filter_out_zero_similarity:
                samples = list(filter(lambda x: x != 0, samples))
            return samples
        print(f'Sampling similarity from scratch.')
        samples = self.draw_random_samples(num_samples=num_of_samples)
        if cache_dir:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            print(f'Saving similarity sampling to file: {file}')
            df = pd.DataFrame(samples, columns=['similarity'])
            df.to_csv(file, index=False)

        if filter_out_zero_similarity:
            samples = list(filter(lambda x: x != 0, samples))
        return samples

    # def _load_similarity_sampling_from_file(self, num_of_samples) -> List[float]:
    #     samples_df = pd.read_csv(os.path.join('dfs', 'samples', str(num_of_samples), f'{self.dataset.name}.csv'))
    #     return samples_df['similarity'].tolist()
