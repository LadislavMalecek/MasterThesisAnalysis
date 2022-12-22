function green {
    printf "\033[92m$@\033[0m\n"
}

#green "Running dataset downloader and transformer for movie_lens"
#poetry run python gather_datasets/download_and_transform.py --datasets movie_lens

green "Running dataset downloader and transformer for movie_lens_small"
poetry run python gather_datasets/download_and_transform.py --datasets movie_lens_small

green "Running dataset downloader and transformer for netflix"
poetry run python gather_datasets/download_and_transform.py --datasets netflix

green "Running dataset downloader and transformer for lastfm"
poetry run python gather_datasets/download_and_transform.py --datasets lastfm

green "Running dataset downloader and transformer for kgrec"
poetry run python gather_datasets/download_and_transform.py --datasets kgrec

green "Running dataset downloader and transformer for spotify"
poetry run python gather_datasets/download_and_transform.py --datasets spotify



# Get the ground truths
green "Running matrix factorization for kgrec"
poetry run python ./matrix_factorization/matrix_factorization.py --rating-type implicit --input ./datasets/kgrec/music_ratings.csv.gz --num-factors 50 --num-iterations 15 --random-seed 42

green "Running matrix factorization for movie_lens_small"
poetry run python ./matrix_factorization/matrix_factorization.py --rating-type explicit --input ./datasets/movie_lens_small/ratings.csv.gz --num-factors 200 --num-iterations 15 --random-seed 42

green "Running matrix factorization for Netflix"
poetry run python ./matrix_factorization/matrix_factorization.py --rating-type explicit --input ./datasets/netflix/ratings.csv.gz --num-factors 300 --num-iterations 15 --random-seed 42

green "Running matrix factorization for Spotify"
poetry run python ./matrix_factorization/matrix_factorization.py --rating-type implicit --input ./datasets/spotify/ratings.csv.gz --num-factors 300 --num-iterations 15 --random-seed 42


# Generate synthetic groups
datasets=(
    ./datasets/kgrec/music_ratings.csv.gz
    ./datasets/movie_lens_small/ratings.csv.gz
    ./datasets/kgrec/music_ratings.csv.gz
    ./datasets/netflix/ratings.csv.gz
)

for dataset in "${datasets[@]}"
do
  poetry run python ./create_groups/create_random_groups.py --group-sizes 4,6,8 --num-groups-to-generate 1000 --input $dataset
  poetry run python ./create_groups/create_topk_groups.py --group-sizes 4,6,8 --num-groups-to-generate 1000 --num-of-candidates 1000 --input $dataset
  poetry run python ./create_groups/create_prs_groups.py --scaling-exponent 1 --group-sizes 4,6,8 --num-groups-to-generate 1000 --num-of-candidates 1000 --input $dataset
  poetry run python ./create_groups/create_prs_groups.py --scaling-exponent 4 --group-sizes 4,6,8 --num-groups-to-generate 1000 --num-of-candidates 1000 --input $dataset
done



datasets=(
   ./datasets/kgrec
   ./datasets/movie_lens_small
   ./datasets/kgrec
   ./datasets/netflix
)


# single run experiments
# iterate through all files in the ./datasets/groups
for dataset in "${datasets[@]}"
do
  # get all files in the directory $dataset/groups
  echo 'Running group recommender for' $file
  poetry run python ./experiments/run_algorithms.py --input-groups-directory $dataset/groups --input_mf $datasets/mf/
done

# create group weights for the weighted experiments
for dataset in "${datasets[@]}"
do
  # get all files in the directory $dataset/groups
  echo 'Creating group weights for' $dataset
  poetry run python ./create_groups/create_group_weights.py --output-dir $dataset/groups/weights
done