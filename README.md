# Marketing media mix modelling
marketing media mix modelling with data generation and bayesian hierarchical structure.

## First things first

You should use a virtual environment with python version 3.11.
Install python packages using
```
pip install -r requirements.txt
```

from configs.py you can set available devices for the jax.

## Data generation
Generate suitable data for yourself by modifying the gendata.py accordingly and running
```
python gendata.py
```

this will create a csv-file of generated data inside 'data/'-folder. It will also dump a bunch of images into 'datagen_images/'-folder

## Linear bayesian algorithm
Bayesian models used can be found inside models.py
Run Bayesian linear algorithm by running 
```
python linear_algo_run.py
```
This will generate images into 'results/'-folder.