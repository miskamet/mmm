import logging
import os
import jax 
import jax.numpy as jnp
import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro import plate
numpyro.set_host_device_count(8)
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

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
    #Base  estimation
    base = numpyro.sample("base", dist.Normal(0,1))
    # j-pot categories
    n_jackpot_categories = len(np.unique(jackpot_category)) # infer number of categories
    with plate("jackpot_plate", n_jackpot_categories):  # Use a plate for categories
        c_jackpot = numpyro.sample('c_jackpot', dist.Normal(0, 1)) # Prior for each category
    #jackpot prior
    jackpot_effect = c_jackpot[jackpot_category]
    jackpot = jackpot_effect*jackpot_category

    # Trend component
    trend_coef = numpyro.sample("trend_coef", dist.Normal(0, 1))  # Prior for the trend
    trend = trend_coef * index

    # Seasonality (using day_of_year as a categorical variable)
    n_days = 366  # Maximum possible days in a year
    with plate("day_of_year_plate", n_days):
        day_of_year_effect = numpyro.sample("day_of_year_effect", dist.Normal(0, 1))  # Seasonality effect for each day

    seasonality = day_of_year_effect[dayofyear -1]  # dayofyear is 1-indexed, so we adjust

    # Coefficiency spends prior
    coef_spends = numpyro.sample('coef_spends', dist.HalfNormal(1))
    media_effect = coef_spends*spends


    # estimate sales
    rev_sigma = numpyro.sample("sigma", dist.Normal(0, 10))
    rev_mean = numpyro.deterministic("mean", base + jackpot_effect*jackpot + media_effect + trend + seasonality)


    sales = numpyro.sample('sales', dist.Normal(rev_mean, rev_sigma), obs=sales)



    