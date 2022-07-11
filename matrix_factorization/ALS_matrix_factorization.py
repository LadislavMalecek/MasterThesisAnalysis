import argparse
import os
from pathlib import Path
import numpy as np

from scipy.sparse import csc_matrix
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import ray
from random import shuffle, random
from more_itertools import chunked


DESCRIPTION = 'Alternating Least Squares (ALS) matrix factorization. Learns latent features for users and movies.'

def print_green(text):
    print('\033[92m' + text + '\033[0m')

def print_yellow(text):
    print('\033[93m' + text + '\033[0m')

def check_and_process_data(data_df):
    assert data_df['user_id'].nunique() == data_df['user_id'].max() + 1, 'user_id must be sequential and start from 0'
    assert data_df['item_id'].nunique() == data_df['item_id'].max() + 1, 'item_id must be sequential and start from 0'

    if 'user_id' not in data_df.columns or 'item_id' not in data_df.columns:
        raise Exception('Dataframe does not have user_id and item_id columns')

    if not 'rating' in data_df.columns:
        data_df['rating'] = 1

    # keep only relevant columns
    data_df = data_df[['user_id', 'item_id', 'rating']]
    return data_df

def split_to_train_test_sparse(data_df, min_n_ratings = 10, test_size=0.2):
    # keep only users with at least n ratings
    num_ratings = data_df.groupby('user_id').size()
    index_size_ok = num_ratings[num_ratings >= min_n_ratings].index
    index_size_low = num_ratings[num_ratings < min_n_ratings].index
    original_data_ok = data_df[data_df['user_id'].isin(index_size_ok)]
    original_data_low = data_df[data_df['user_id'].isin(index_size_low)]

    training, testing = train_test_split(original_data_ok, test_size=0.2, stratify=original_data_ok['user_id'])
    training = pd.concat([training, original_data_low])

    num_of_users = data_df['user_id'].max() + 1
    num_of_items = data_df['item_id'].max() + 1

    # transform to sparse matricies, optimized for reading by column
    training_data_csc = csc_matrix((training['rating'], (training['user_id'], training['item_id'])), shape=(num_of_users, num_of_items))
    training_data_t_csc = csc_matrix((training['rating'], (training['item_id'], training['user_id'])), shape=(num_of_items, num_of_users))
    testing_data_csc = csc_matrix((testing['rating'], (testing['user_id'], testing['item_id'])), shape=(num_of_users, num_of_items))
    
    # check if the last value is only in training data xor testing data
    uid = int(original_data_ok.iloc[-1]['user_id'])
    iid = int(original_data_ok.iloc[-1]['item_id'])
    rating = original_data_ok.iloc[-1].rating

    in_train = training_data_csc[uid, iid] == rating
    in_test = testing_data_csc[uid, iid] == rating
    assert (in_train and not in_test) or (not in_train and in_test), 'Dataframe is not split correctly'

    return training_data_csc, training_data_t_csc, testing_data_csc

@ray.remote
def process_single_index_ray(j, l2_lambda, data_ray, fixed_matrix_ray):
    nonzeros = data_ray[:,j].nonzero()[0]
    y = data_ray[nonzeros, j].todense()
    X = fixed_matrix_ray[:, nonzeros]
    return np.squeeze(np.linalg.inv(X.dot(X.T) + l2_lambda * np.eye(X.shape[0])).dot(X.dot(y)))

def cf_ridge_regression(target_matrix, fixed_matrix, data_ray, l2_lambda):
    '''
    Solves a ridge regression problem using a closed form solution:
        w_i = (X'X + lambda * I)^-1 X'y
    for all i in the target matrix.
    '''
    # put the fixed matrix to shared ray storage
    fixed_matrix_ray = ray.put(fixed_matrix)

    # we want to shufle the indices of the matrix to get more consistent tqdm progress
    # if we don't shuffle, and the data are sorted by the amount of ratings per user
    # then the proggress will start slow and then speed up
    # this could lead to cache underutilization, but no impact was observed
    matrix_column_indexes = list(range(target_matrix.shape[1]))
    shuffle(matrix_column_indexes, )

    with tqdm(total=target_matrix.shape[1]) as pbar:
        for chunk_j in chunked(matrix_column_indexes, n=100):
            futures = []
            for j in chunk_j:
                futures.append(process_single_index_ray.remote(j, l2_lambda, data_ray, fixed_matrix_ray))
            results = ray.get(futures)

            for j, result in zip(chunk_j, results):
                target_matrix[:,j] = result
            pbar.update(len(chunk_j))

    # remove the object from shared ray storage
    del fixed_matrix_ray

def sum_squared_error_on_subset(gt_data, U, M, subset_size = 1000):
    U_small = U[:,0:subset_size].T
    M_small = M[:,0:subset_size]
    resulting = U_small.dot(M_small)
    gt_data_subset = gt_data[0:subset_size,0:subset_size]
    diff = gt_data_subset - resulting
    error = diff[gt_data_subset[0:subset_size,0:subset_size].nonzero()]
    total_error = (np.array(error) ** 2).sum()
    return total_error

def train(data, num_features, convergence_threshold, l2_lambda, total_ratings, num_iterations):
    # unpack data
    training_data_csc, training_data_t_csc, testing_data_csc = data

    num_users = training_data_csc.shape[0]
    num_items = training_data_csc.shape[1]

    U_features = np.ones((num_features, num_users)) # k x u
    I_features = np.ones((num_features, num_items)) # k x m

    training_trace = []
    testing_trace = []

    train_error_delta = convergence_threshold + 1.
    current_error = 1.
    current_step = 0

    while train_error_delta > convergence_threshold and current_step < num_iterations:
        print_green(f'Iteration {current_step + 1}/{num_iterations}')
        # Use the closed-form solution for the ridge-regression subproblems
        training_data_csc_ray = ray.put(training_data_csc)
        training_data_t_csc_ray = ray.put(training_data_t_csc)
        print('Fitting M')
        cf_ridge_regression(target_matrix=I_features, fixed_matrix=U_features, data_ray=training_data_csc_ray, l2_lambda=l2_lambda)
        print('Fitting U')
        cf_ridge_regression(target_matrix=U_features, fixed_matrix=I_features, data_ray=training_data_t_csc_ray, l2_lambda=l2_lambda)

        # Track performance in terms of RMSE
        train_error = sum_squared_error_on_subset(training_data_csc, U_features, I_features)
        test_error = sum_squared_error_on_subset(testing_data_csc, U_features, I_features)
        training_trace.append(np.sqrt(train_error / total_ratings))
        testing_trace.append(np.sqrt(test_error / total_ratings))

        # Track convergence
        previous_error = current_error
        current_error = train_error
        train_error_delta = np.abs(previous_error - current_error) / (previous_error + convergence_threshold)
        print_yellow(f'Training error: {np.sqrt(train_error / total_ratings)},'
            f' Testing error: {np.sqrt(test_error / total_ratings)},'
            f' Train error delta: {train_error_delta}')
        # Update the step counter
        current_step += 1
    
    if current_step == num_iterations:
        print_green('Maximum number of iterations reached')
    else:
        print_green('Converged in {0} iterations'.format(current_step))

    return U_features, I_features, training_trace, testing_trace


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--input', default='./datasets/kgrec/music_ratings.csv.gz', help='The dataset to use, needs to be csv dataframe with columns "user_id", "item_id" and optionally "rating".')
    parser.add_argument('--output_dir', default=None, help='The directory where the resulting matricies will be saved. Default is "mf" dir under the input data directory.')
    parser.add_argument('--num_factors', type=int, default=200, help='The number of latent factors to use.')
    parser.add_argument('--num_iterations', type=int, default=100, help='The number of iterations to use.')
    parser.add_argument('--regularization', type=float, default=0.1, help='The regularization parameter.')
    parser.add_argument('--convergence_threshold', type=float,  default=0.001, help='The convergence threshold.')
    parser.add_argument('--num_threads', default=None, help='The number of threads to use. Default is system cores - 2')
    parser.add_argument('--random_seed', default=42, help='The random seed to use. Default is None')
    
    args = parser.parse_args()

    # if args.random_seed is not None:
    #     random.seed(args.random_seed)

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input), 'mf')

    if args.num_threads is None:
        args.num_threads = os.cpu_count() - 2

    # Load data from file
    print('Loading data from {0}'.format(args.input))
    data_df = pd.read_csv(args.input)
    print('Checking and processing data')
    check_and_process_data(data_df)
    print('Spitting data into training and testing sets and creating efficient sparse matrices')
    data = split_to_train_test_sparse(data_df)

    print('Initializing Ray with {0} threads'.format(args.num_threads))
    ray.init(num_cpus=args.num_threads)

    # Train the model
    print('Training model...')
    U_features, I_features, training_trace, testing_trace = train(
        data=data,num_features=args.num_factors,
        convergence_threshold=args.convergence_threshold,
        l2_lambda=args.regularization,
        total_ratings=len(data_df),
        num_iterations=args.num_iterations)

    print('Saving user and item features in numpy datafile (.npy) to {0}'.format(args.output_dir))
    Path.mkdir(Path(args.output_dir), exist_ok=True)
    np.save(os.path.join(args.output_dir, 'U_features.npy'), U_features)
    np.save(os.path.join(args.output_dir, 'I_features.npy'), I_features)

    print('Saving training and testing traces in csv file to {0}'.format(args.output_dir))
    trace_df = pd.DataFrame({'training': training_trace, 'testing': testing_trace})
    trace_df.to_csv(os.path.join(args.output_dir, 'trace.csv'), index=False)

    print_green('Done!')