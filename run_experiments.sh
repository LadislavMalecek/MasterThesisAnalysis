function green {
    printf "\033[92m$@\033[0m\n"
}

# Get the 
green "Running matrix factorization for kgrec"
poetry run python ./matrix_factorization/ALS_matrix_factorization.py --input ./datasets/kgrec/music_ratings.csv.gz --num_factors 50 --num_iterations 100

green "Running matrix factorization for MovieLens"
poetry run python ./matrix_factorization/ALS_matrix_factorization.py --input ./datasets/movie_lens/ratings.csv.gz --num_factors 200 --num_iterations 100

green "Running matrix factorization for Netflix"
poetry run python ./matrix_factorization/ALS_matrix_factorization.py --input ./datasets/netflix/ratings.csv.gz --num_factors 300 --num_iterations 100

commented out temporarily due to a slow running time
green "Running matrix factorization for Spotify"
poetry run python ./matrix_factorization/ALS_matrix_factorization.py --input ./datasets/spotify/ratings.csv.gz --num_factors 300 --num_iterations 100