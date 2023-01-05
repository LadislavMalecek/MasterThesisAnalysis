# Master thesis - experiments and implementation

**Title:** Fairness in group recommender systems

**Author:** Bc. Ladislav Malecek

**Supervisor:** Mgr. Ladislav Peska, Ph.D.

**Requirements:** Python version and libraries that are required are described in `pyproject.toml`

---

## Reproducibility receipe

1. Install the required dependencies
We are using python 3.9 and poetry as a dependency manager. Install poetry and then run `poetry install` to install all the dependencies. If you are not using poetry, you can install the dependencies manually.

2. Run our reproducibility script
```
poetry run python run_experiments.py
```
---
Repository structure of the most important files and folders:
```
.
├── create_groups
│   ├── create_prs_groups.py
│   ├── create_random_groups.py
│   └── create_topk_groups.py
│
├── gather_datasets
│   └── download_and_transform.py
│
├── evaluation
│   ├── evaluation_longterm.ipynb
│   ├── evaluation_uniform.ipynb
│   └── evaluation_weighted.ipynb
│
├── experiments
│   ├── run_longterm_algorithms.py
│   ├── run_uniform_algorithms.py
│   └── run_weighted_algorithms.py
│
├── matrix_factorization
│   └── matrix_factorization.py
│
└── run_experiments.sh
```

---

## Evaluation steps
The evaluation has 5 parts:
1. Gather datasets
2. Create artificial groups
3. Calculate recommendation ratings that serve as the ground truth.
4. Run recommendation algorithms
5. Evaluate results

For more information to any mentioned python script, available arguments, and defaults settings, run `python <script> --help`

## 1. Gather Datasets:

We have created an automatic tool for downloading, cleaning and proccessing the required datasets. Preffered variant would be to have the datasets already clean and ready in this repository, or hosted somewhere else, but that is not possible due to the datasets' licencing.

Run 
```
poetry run python gather_datasets/download_and_transform.py
```

This python script has multiple options to make it convinient and reusable for different projects as well. You can specify which of the supported datasets will be downloaded, if they will be compressed and if they will be only stored or as well cleaned and processed and where the results will be stored.


## 2. Create artificial groups:

We have created an automatic tool for creating artificial groups.


For creation of PRS groups run
```
poetry run python create_groups/create_prs_groups.py
```

For creation of random groups run
```
poetry run python create_groups/create_random_groups.py
```

and for creation of top-k groups run
```
poetry run python create_groups/create_topk_groups.py
```



## 3. Calculate recommendation ratings:

We use fast and conviniently parallelizable algorithm called 'Alternating Least Squares (ALS) matrix factorization' to calculate the ground truths. The implementation for explicit and implicit datasets differs, for explicit datasets we are using our own implementation, and for implicit datasets we are using the implementation from the `implicit` library.

Run
```
poetry run python ./matrix_factorization/matrix_factorization.py
```

## 4. Run recommendation algorithms:
For each scenario we have created a python script that runs the algorithms and saves the results to the `results` folder.
Run the scripts in the `experiments` folder to run the algorithms.

## 5. Evaluate results:

Run the jupyter notebooks in the `evaluation` folder to evaluate the results.