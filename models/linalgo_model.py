import logging
import os
import jax 
import jax.numpy as jnp
import numpy as np

from configs import config
from utils import GeometricAdstockTransformer,BetaHillTransformation

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro import plate

devices = config.devices


numpyro.set_host_device_count(devices)
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={devices}'

def bayes_linreg_model(jackpot_category,spends, sales, dayofyear, index):
    '''
    Bayesian linear regression model for marketing media mix.
    ---------------------------------------------------------
    Inputs:
    jackpot_category: jnp.array
        array of jackpot sizes
    coef_spends: jnp.array
        array of media costs
    sales: jnp.array
        array of radio spends
    ---------------------------------------------------------
    output: numpyro model
    '''
    # !!!!! PRIOR DISTRIBUTIONS !!!!!!
    # base
    base = numpyro.sample("base", dist.Normal(0,1))
    # j-pot priors
    n_jackpot_categories = len(np.unique(jackpot_category)) # infer number of categories
    with plate("jackpot_plate", n_jackpot_categories):  # Use a plate for categories
        c_jackpot = numpyro.sample('c_jackpot', dist.Normal(0, 10)) # Prior for each category

    #trend
    trend_coef = numpyro.sample("trend_coef", dist.Normal(0, 1))

    # Seasonality priors
    n_days = 366  # Maximum possible days in a year
    with plate("day_of_year_plate", n_days):  # Plate goes outside
        seasonality_effect = numpyro.sample("seasonality_effect", dist.Normal(0, 1))
    
    # Media spend prior
    coef_spends = numpyro.sample('media_coef', dist.Normal(jnp.mean(spends)/1000,jnp.std(spends)/500))
    # Error prior
    epsilon = numpyro.sample("epsilon", dist.Normal(0, 0.05))

    #!!! COMPONENT PARAMETERS !!!
    # Trend component

    trend = numpyro.deterministic("trend",trend_coef * index)

    # Seasonality
    seasonality = numpyro.deterministic('seasonality', seasonality_effect[dayofyear - 1])
     # dayofyear is 1-indexed, so we adjust

    # Coefficiency spends prior
    #TODO: add geometric transformation and beta-hill transformation for media cost data
    media_effect = numpyro.deterministic("media_effect",coef_spends*spends)

    # Jackpot

    jackpot_effect = c_jackpot[jackpot_category]
    jackpot = numpyro.deterministic("jackpot",jackpot_effect*jackpot_category)


    # estimate sales
    rev_sigma = numpyro.sample("sigma", dist.Uniform(0, 1))
    rev_mean = numpyro.deterministic("mean", base + jackpot + spends + media_effect + trend + seasonality + epsilon)


    sales = numpyro.sample('sales', dist.Normal(rev_mean, rev_sigma), obs=sales)




#TODO: Create a model where jackpot level is a confounder, effecting both sales and media channels 