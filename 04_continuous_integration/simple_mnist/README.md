mlops_02
==============================

Cookiecutter structured version of exercises from day 1.

`src/make_dataset.py` downloads and transforms the MNIST dataset. The files are saved to `data/processes` while the raw 
data has been manually moved to `data/raw` from the `data/processed/MNIST` folder.

The model is defined in the `model.py` file.

`src/models/main.py` acts as a driver script to both train and evaluate the model. This is done by adding the keywords `train` or `evaluate`. E.g.: `python main.py train` or `python main.py evaluate`.
When training, the flag `--lr=` can be specified to define the learning rate (default `3e-4`). `--num_epochs` specifies the total number of epoch for training (default `10`).
Upon completion a checkpoint is saved as `models/checkpoint.pth`. The training learning curve per batch and per epoch is saved under `reports/figures/01_final_exercise_training_losses.png`.
When evaluating, the `--load_model_from=` flag specifies which model to load (default `models/checkpoint.pth`). The accuracy is printed when finished.

`src/models/predict_model.py` computes and predicts the labels of an either a folder of images or a pickle-file. Currently, 10 random samples are generated for demonstration purposes.

`src/visualization/visualize.py` computes a t-SNE embedding of the 128 first test images (easily adjustable) as extracted from the penultimate layer of the trained model. The output is saved as `reports/figures/02_tSNE.png`.





Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
