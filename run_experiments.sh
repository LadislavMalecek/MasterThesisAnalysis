function green {
    printf "\033[92m$@\033[0m\n"
}

# -------------------------------------------------------
# gather datasets
# -------------------------------------------------------
green "Running dataset downloader and transformer for kgrec"
poetry run python gather_datasets/download_and_transform.py --datasets kgrec

green "Running dataset downloader and transformer for movie_lens"
poetry run python gather_datasets/download_and_transform.py --datasets movie_lens

green "Running dataset downloader and transformer for movie_lens_small"
poetry run python gather_datasets/download_and_transform.py --datasets movie_lens_small

green "Running dataset downloader and transformer for netflix"
poetry run python gather_datasets/download_and_transform.py --datasets netflix

# green "Running dataset downloader and transformer for lastfm"
# poetry run python gather_datasets/download_and_transform.py --datasets lastfm

green "Running dataset downloader and transformer for spotify"
poetry run python gather_datasets/download_and_transform.py --datasets spotify



# -------------------------------------------------------
# Get the ground truths
# -------------------------------------------------------
green "Running matrix factorization for kgrec"
poetry run python ./matrix_factorization/matrix_factorization.py --rating-type implicit --input ./datasets/kgrec/music_ratings.csv.gz --num-factors 50 --num-iterations 15 --random-seed 42

green "Running matrix factorization for movie_lens_small"
poetry run python ./matrix_factorization/matrix_factorization.py --rating-type explicit --input ./datasets/movie_lens_small/ratings.csv.gz --num-factors 200 --num-iterations 15 --random-seed 42

green "Running matrix factorization for movie_lens"
poetry run python ./matrix_factorization/matrix_factorization.py --rating-type explicit --input ./datasets/movie_lens/ratings.csv.gz --num-factors 200 --num-iterations 15 --random-seed 42

green "Running matrix factorization for netflix"
poetry run python ./matrix_factorization/matrix_factorization.py --rating-type explicit --input ./datasets/netflix/ratings.csv.gz --num-factors 300 --num-iterations 15 --random-seed 42

green "Running matrix factorization for spotify"
poetry run python ./matrix_factorization/matrix_factorization.py --rating-type implicit --input ./datasets/spotify/ratings.csv.gz --num-factors 300 --num-iterations 15 --random-seed 42



#-------------------------------------------------------
# Generate synthetic groups
# -------------------------------------------------------
datasets=(
    ./datasets/kgrec/music_ratings.csv.gz
    ./datasets/movie_lens_small/ratings.csv.gz
    ./datasets/movie_lens/ratings.csv.gz
    ./datasets/spotify/ratings.csv.gz
    ./datasets/netflix/ratings.csv.gz
)
for dataset in "${datasets[@]}"
do
  poetry run python ./create_groups/create_random_groups.py --group-sizes 2,3,4,6,8 --num-groups-to-generate 1000 --input $dataset
  poetry run python ./create_groups/create_topk_groups.py --group-sizes 2,3,4,6,8 --num-groups-to-generate 1000 --num-of-candidates 1000 --input $dataset
  poetry run python ./create_groups/create_prs_groups.py --scaling-exponent 1 --group-sizes 2,3,4,6,8 --num-groups-to-generate 1000 --num-of-candidates 1000 --input $dataset
  poetry run python ./create_groups/create_prs_groups.py --scaling-exponent 4 --group-sizes 2,3,4,6,8 --num-groups-to-generate 1000 --num-of-candidates 1000 --input $dataset
done



datasets=(
   ./datasets/kgrec
   ./datasets/movie_lens_small
   ./datasets/movie_lens
   ./datasets/spotify
   ./datasets/netflix
)

# -------------------------------------------------------
# create group weights for the weighted experiments
# -------------------------------------------------------
for dataset in "${datasets[@]}"
do
  # get all files in the directory $dataset/groups
  echo 'Creating group weights for' $dataset
  poetry run python create_groups/create_group_weights.py --group-sizes 2,3,4,6,8 --output-dir $dataset/groups/weights
done

datasets=(
   ./datasets/kgrec
   ./datasets/movie_lens_small
   ./datasets/movie_lens
   ./datasets/spotify
   ./datasets/netflix
)

# -------------------------------------------------------
# run experiments
# -------------------------------------------------------
for dataset in "${datasets[@]}"
do
  # get all files in the directory $dataset/groups
  echo 'Running group recommenders for' $file
  poetry run python experiments/run_uniform_algorithms.py --group-sizes 2,3,4,6,8 --input-groups-directory $dataset/groups --input-mf $dataset/mf/
  poetry run python experiments/run_weighted_algorithms.py --group-sizes 2,3,4,6,8 --input-groups-directory $dataset/groups --input-mf $dataset/mf/
  poetry run python experiments/run_longterm_algorithms.py --group-sizes 2,3,4,6,8 --input-groups-directory $dataset/groups --input-mf $dataset/mf/
done


# -------------------------------------------------------
# run evaluation
# -------------------------------------------------------
echo 'To run the evaluation, please run the following notebooks:'
echo '1. ./evaluation/evaluation_uniform.ipynb'
echo '2. ./evaluation/evaluation_weighted.ipynb'
echo '3. ./evaluation/evaluation_longterm.ipynb'