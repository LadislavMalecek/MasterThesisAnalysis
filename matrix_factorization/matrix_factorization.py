import sys
from os import path
sys.path.append(path.join(sys.path[0], '..'))

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import implicit
import numpy as np

from scipy.sparse import csc_matrix, csr_matrix
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import ray
from random import shuffle, random
from more_itertools import chunked


class ALSMatrixFactorization:
    def __init__(self, num_features: int):
        self.num_features = num_features
        self.i_features = None
        self.u_features = None

        self.training_trace = []
        self.testing_trace = []

    @staticmethod
    def _print_green(text: str):
        print('\033[92m' + text + '\033[0m')

    @staticmethod
    def _print_yellow(text: str):
        print('\033[93m' + text + '\033[0m')

    @staticmethod
    @ray.remote
    def _process_single_index_ray(j: int, l2_lambda: float, data_ray, fixed_matrix_ray):
        nonzeros = data_ray[:, j].nonzero()[0]
        y = data_ray[nonzeros, j].todense()
        X = fixed_matrix_ray[:, nonzeros]
        return np.squeeze(np.linalg.inv(X.dot(X.T) + l2_lambda * np.eye(X.shape[0])).dot(X.dot(y)))

    def _cf_ridge_regression(self, target_matrix: np.ndarray, fixed_matrix: np.ndarray, data_ray, l2_lambda: float):
        """
        Solves a ridge regression problem using a closed form solution:
            w_i = (X'X + lambda * I)^-1 X'y
        for all i in the target matrix.

        target_matrix:

        """
        # put the fixed matrix to shared ray storage
        fixed_matrix_ray = ray.put(fixed_matrix)

        # we want to shuffle the indices of the matrix to get more consistent tqdm progress
        # if we don't shuffle, and the data are sorted by the amount of ratings per user
        # then the progress will start slow and then speed up
        # this could lead to cache under-utilization, but no impact was observed
        matrix_column_indexes = list(range(target_matrix.shape[1]))
        shuffle(matrix_column_indexes, )

        with tqdm(total=target_matrix.shape[1]) as pbar:
            for chunk_j in chunked(matrix_column_indexes, n=100):
                futures = []
                for j in chunk_j:
                    futures.append(self._process_single_index_ray.remote(j, l2_lambda, data_ray, fixed_matrix_ray))
                results = ray.get(futures)

                for j, result in zip(chunk_j, results):
                    target_matrix[:, j] = result
                pbar.update(len(chunk_j))

        # remove the object from shared ray storage
        del fixed_matrix_ray

    @staticmethod
    def _sum_squared_error_on_subset(ground_truth_data, u_factors, i_factors, subset_size=1000):
        u_factors_small = u_factors[:, 0:subset_size].T
        i_factors_small = i_factors[:, 0:subset_size]
        resulting = u_factors_small.dot(i_factors_small)
        gt_data_subset = ground_truth_data[0:subset_size, 0:subset_size]
        diff = gt_data_subset - resulting
        error = diff[gt_data_subset[0:subset_size, 0:subset_size].nonzero()]
        total_error = (np.array(error) ** 2).sum()
        return total_error

    def _init_features(self, n_users: int, n_items: int, random_seed: Optional[int]) -> None:
        if random_seed:
            np.random.seed(random_seed)
        self.u_features = np.random.uniform(0, 0.1, size=(self.num_features, n_users))
        self.i_features = np.random.uniform(0, 0.1, size=(self.num_features, n_items))

    def train(
            self,
            training_data_user_item_csc: csc_matrix,
            *,
            testing_data_user_item_csc: csc_matrix = None,
            convergence_threshold: float = 0.001,
            l2_lambda: float = 0.1,
            num_iterations: int = 15,
            num_threads: Optional[int] = None,
            random_seed: Optional[int] = None,
    ) -> None:
        if not ray.is_initialized():
            if num_threads is None:
                num_threads = os.cpu_count()
            print('Initializing Ray with {0} threads'.format(num_threads))
            ray.init(num_cpus=num_threads)

        num_users = training_data_user_item_csc.shape[0]
        num_items = training_data_user_item_csc.shape[1]

        if self.u_features is None or self.i_features is None:
            self._init_features(num_users, num_items, random_seed)

        training_data_user_item_t_csc = training_data_user_item_csc.T

        train_error_delta = convergence_threshold + 1.
        current_error = 1.
        current_step = 0

        training_data_csc_ray = ray.put(training_data_user_item_csc)
        training_data_t_csc_ray = ray.put(training_data_user_item_t_csc)

        number_of_ratings = training_data_user_item_csc.nnz
        if testing_data_user_item_csc is not None:
            number_of_ratings += testing_data_user_item_csc.nnz

        while train_error_delta > convergence_threshold and current_step < num_iterations:
            self._print_green(f'Iteration {current_step + 1}/{num_iterations}')
            # Use the closed-form solution for the ridge-regression subproblems
            print('Fitting M')
            self._cf_ridge_regression(target_matrix=self.i_features, fixed_matrix=self.u_features, data_ray=training_data_csc_ray, l2_lambda=l2_lambda)
            print('Fitting U')
            self._cf_ridge_regression(target_matrix=self.u_features, fixed_matrix=self.i_features, data_ray=training_data_t_csc_ray, l2_lambda=l2_lambda)

            # Track performance in terms of RMSE
            train_error = self._sum_squared_error_on_subset(training_data_user_item_csc, self.u_features, self.i_features)
            self.training_trace.append(np.sqrt(train_error / number_of_ratings))

            test_error = None
            if testing_data_user_item_csc is not None:
                test_error = self._sum_squared_error_on_subset(testing_data_user_item_csc, self.u_features, self.i_features)
                self.testing_trace.append(np.sqrt(test_error / number_of_ratings))

            # Track convergence
            previous_error = current_error
            current_error = train_error
            train_error_delta = np.abs(previous_error - current_error) / (previous_error + convergence_threshold)

            progress_text = f'Training error: {np.sqrt(train_error / number_of_ratings)}, Train error delta: {train_error_delta}'
            if test_error:
                progress_text += f' Testing error: {np.sqrt(test_error / number_of_ratings)},'

            self._print_yellow(progress_text)
            # Update the step counter
            current_step += 1

        if current_step == num_iterations:
            self._print_green('Maximum number of iterations reached')
        else:
            self._print_green('Converged in {0} iterations'.format(current_step))


def check_and_process_data(data_df: pd.DataFrame) -> pd.DataFrame:
    assert data_df['user_id'].nunique() == data_df['user_id'].max() + 1, 'user_id must be sequential and start from 0'
    assert data_df['item_id'].nunique() == data_df['item_id'].max() + 1, 'item_id must be sequential and start from 0'

    if 'user_id' not in data_df.columns or 'item_id' not in data_df.columns:
        raise Exception('Dataframe does not have user_id and item_id columns')

    if not 'rating' in data_df.columns:
        data_df['rating'] = 1

    # keep only relevant columns
    data_df = data_df[['user_id', 'item_id', 'rating']]
    return data_df


def split_to_train_test_sparse(data_df: pd.DataFrame, min_n_ratings: int = 10, test_size: float = 0.2) -> Tuple[csc_matrix, csc_matrix]:
    # keep only users with at least n ratings
    num_ratings = data_df.groupby('user_id').size()
    index_size_ok = num_ratings[num_ratings >= min_n_ratings].index
    index_size_low = num_ratings[num_ratings < min_n_ratings].index
    original_data_ok = data_df[data_df['user_id'].isin(index_size_ok)]
    original_data_low = data_df[data_df['user_id'].isin(index_size_low)]

    training, testing = train_test_split(original_data_ok, test_size=test_size, stratify=original_data_ok['user_id'])
    training = pd.concat([training, original_data_low])

    num_of_users = data_df['user_id'].max() + 1
    num_of_items = data_df['item_id'].max() + 1

    # transform to sparse matricies, optimized for reading by column
    training_data_csc = csc_matrix((training['rating'], (training['user_id'], training['item_id'])), shape=(num_of_users, num_of_items))
    testing_data_csc = csc_matrix((testing['rating'], (testing['user_id'], testing['item_id'])), shape=(num_of_users, num_of_items))

    # check if the last value is only in training data xor testing data
    uid = int(original_data_ok.iloc[-1]['user_id'])
    iid = int(original_data_ok.iloc[-1]['item_id'])
    rating = original_data_ok.iloc[-1].rating

    in_train = training_data_csc[uid, iid] == rating
    in_test = testing_data_csc[uid, iid] == rating
    assert (in_train and not in_test) or (not in_train and in_test), 'Dataframe is not split correctly'

    return training_data_csc, testing_data_csc


def main_explicit(
        input_file: str,
        output_dir: str,
        factors: int,
        convergence_threshold: float,
        regularization: float,
        num_iterations: int,
        num_threads: int,
        random_seed: Optional[int] = None,
) -> None:
    # Load data from file
    print('Loading data from {0}'.format(args.input))
    data_df = pd.read_csv(input_file)
    print('Checking and processing data')
    data_df = check_and_process_data(data_df)
    print('Spitting data into training and testing sets and creating efficient sparse matrices')
    # split to training and testing sets
    training_data_csc, testing_data_csc = split_to_train_test_sparse(data_df)

    # init model
    print('Initializing model')
    model = ALSMatrixFactorization(factors)

    # Train the model
    print('Training model...')
    model.train(
        training_data_user_item_csc=training_data_csc,
        testing_data_user_item_csc=testing_data_csc,
        convergence_threshold=convergence_threshold,
        l2_lambda=regularization,
        num_iterations=num_iterations,
        num_threads=num_threads,
        random_seed=random_seed,
    )

    print('Saving user and item features in numpy datafile (.npy) to {0}'.format(output_dir))
    Path.mkdir(Path(output_dir), exist_ok=True)
    np.save(os.path.join(output_dir, 'U_features.npy'), model.u_features)
    np.save(os.path.join(output_dir, 'I_features.npy'), model.i_features)

    print('Saving training and testing traces in csv file to {0}'.format(output_dir))
    trace_df = pd.DataFrame({'training': model.training_trace, 'testing': model.testing_trace})
    trace_df.to_csv(os.path.join(output_dir, 'trace.csv'), index=False)

    print('Done!')


def main_implicit(
        input_file: str,
        output_dir: str,
        factors: int,
        regularization: float,
        num_iterations: int,
        num_threads: int,
        random_seed: Optional[int] = None,
) -> None:
    # Load data from file
    print('Loading data from {0}'.format(args.input))
    data_df = pd.read_csv(input_file)
    print('Checking and processing data')
    check_and_process_data(data_df)

    # train the model on a sparse matrix of user/item/confidence weights

    # we need crs matrix for implicit library as specified in the documentation
    # user_items: csr_matrix
    #   Matrix of confidences for the liked items. This matrix should be a csr_matrix where
    #   the rows of the matrix are the user, the columns are the items liked by that user,
    #   and the value is the confidence that the user liked the item.
    print('Initializing model')
    num_of_users = data_df['user_id'].max() + 1
    num_of_items = data_df['item_id'].max() + 1
    data_indices = (data_df['user_id'], data_df['item_id'])  # row, col
    user_item_data_csr = csr_matrix((data_df['rating'], data_indices), shape=(num_of_users, num_of_items))

    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=num_iterations,
        calculate_training_loss=True,
        num_threads=num_threads,
        random_state=random_seed,
    )

    # Train the model
    print('Training model...')
    model.fit(user_item_data_csr, show_progress=True)

    print('Saving user and item features in numpy datafile (.npy) to {0}'.format(output_dir))
    Path.mkdir(Path(output_dir), exist_ok=True)
    np.save(os.path.join(output_dir, 'U_features.npy'), model.user_factors)
    np.save(os.path.join(output_dir, 'I_features.npy'), model.item_factors)
    print('Done!')


DESCRIPTION = 'Alternating Least Squares (ALS) matrix factorization. Learns latent features for users and movies.'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--rating-type', choices={"implicit", "explicit"}, required=True, help='')
    parser.add_argument('--input', required=True, help='The dataset to use, needs to be csv dataframe with columns "user_id", "item_id" and optionally "rating".')
    parser.add_argument('--output-dir', type=Optional[str], default=None, help='The directory where the resulting matricies will be saved. Default is "mf" dir under the input data directory.')
    parser.add_argument('--num-factors', type=int, required=True, help='The number of latent factors to use.')
    parser.add_argument('--num-iterations', type=int, default=15, help='The number of iterations to use.')
    parser.add_argument('--regularization', type=float, default=0.1, help='The regularization parameter.')
    parser.add_argument('--convergence-threshold', type=float, default=0.001, help='The convergence threshold. Only for explicit algo.')
    parser.add_argument('--num-threads', default=None, help='The number of threads to use. Default is system cores - 2')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed that will be used for the user and item factor matrices')

    args = parser.parse_args()
    print(args)

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input), 'mf')

    if args.num_threads is None:
        args.num_threads = os.cpu_count() - 2

    if args.rating_type == 'explicit':
        main_explicit(
            input_file=args.input,
            output_dir=args.output_dir,
            factors=args.num_factors,
            convergence_threshold=args.convergence_threshold,
            regularization=args.regularization,
            num_iterations=args.num_iterations,
            num_threads=args.num_threads,
            random_seed=args.random_seed,
        )
    elif args.rating_type == 'implicit':
        main_implicit(
            input_file=args.input,
            output_dir=args.output_dir,
            factors=args.num_factors,
            regularization=args.regularization,
            num_iterations=args.num_iterations,
            num_threads=args.num_threads,
            random_seed=args.random_seed,
        )
    else:
        raise ValueError('Invalid rating_type')
