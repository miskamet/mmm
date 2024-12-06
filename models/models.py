import logging
import os
import jax 
import jax.numpy as jnp
import numpy as np

from configs import config
from utils import GeometricAdstockTransformer, LogisticSaturationTransformer, BetaHillTransformation, jackpotgenerator, fourier_modes, trend_transform


import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import TransformReparam
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
    # j-pot priors
    n_jackpot_categories = len(np.unique(jackpot_category)) # infer number of categories

    #trend
    trend_coef = numpyro.sample("trend_coef", dist.Normal(0, 4))

    # Seasonality priors

    
    # Media spend prior
    media_coef = numpyro.sample('media_coef', dist.HalfNormal(2))

    #Normal priors
    alpha= numpyro.sample('alpha', dist.Gamma(2,3))
    l_max = 14
    lam = numpyro.sample('lam', dist.Gamma(2,3))
    # Error prior


    #!!! COMPONENT PARAMETERS !!!
    # Trend component
    seasonality_effect = numpyro.sample("seasonality_effect", dist.Uniform(0, 10))
    # Seasonality
    n_days = 365  # Maximum possible days in a year
    


    geometric_adstock=GeometricAdstockTransformer(alpha=alpha, l=l_max)
    logistic_saturation=LogisticSaturationTransformer(mu=lam) 
    # Jackpot
    with plate("jackpot_plate", n_jackpot_categories):
        c_jackpot = numpyro.sample('c_jackpot', dist.Uniform(-2,2)) 
        jackpot_effect = c_jackpot[jackpot_category]
        jackpot = numpyro.deterministic("jackpot",jnp.dot(jackpot_effect,jackpot_category))

    rev_sigma = numpyro.sample("sigma", dist.Uniform(0, 5))
    # Media spend
    with plate("date_plate", len(index)):
        spends = numpyro.sample("spends", dist.Normal(0,1), obs=spends)
        #media_adstock = numpyro.deterministic("media_adstock", geometric_adstock.transform(X = spends))
        #media_saturated = numpyro.deterministic("media_saturated", logistic_saturation.transform(X = media_adstock))
        media_effect = numpyro.deterministic("media_effect", jnp.dot(media_coef, spends))

        trend = numpyro.deterministic("trend", trend_transform(trend_coef))
        seasonality = numpyro.deterministic("seasonality", fourier_modes(seasonality_effect))
        rev_mean = numpyro.deterministic("mean", jackpot + media_effect + trend + seasonality)
        sales = numpyro.sample("sales", dist.StudentT(rev_mean, rev_sigma), obs=sales)






#TODO: Create a model where jackpot level is a confounder, effecting both sales and media channels 
def bayes_confounder_model(jackpot_category,spends, sales, dayofyear, index):
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
    # j-pot priors
    n_jackpot_categories = len(np.unique(jackpot_category)) # infer number of categories
    with plate("jackpot_plate", n_jackpot_categories):  # Use a plate for categories
        c_jackpot = numpyro.sample('c_jackpot', dist.Normal(0, 10)) # Prior for each category
        jackpot_media_effect = numpyro.sample('jackpot_media_effect', dist.Normal(0, 10))

    #trend
    trend_coef = numpyro.sample("trend_coef", dist.Normal(0, 4))

    # Seasonality priors

    
    # Media spend prior


    #Normal priors
    alpha= numpyro.sample('alpha', dist.Gamma(2,3))
    l_max = 14
    lam = numpyro.sample('lam', dist.Gamma(2,3))
    # Error prior


    #!!! COMPONENT PARAMETERS !!!
    # Trend component
    trend = numpyro.deterministic("trend", trend_transform(trend_coef))
    seasonality_effect = numpyro.sample("seasonality_effect", dist.Uniform(0, 10))
    # Seasonality
    n_days = 365  # Maximum possible days in a year
    seasonality = numpyro.deterministic("seasonality", fourier_modes(seasonality_effect))


    geometric_adstock=GeometricAdstockTransformer(alpha=alpha, l=l_max)
    logistic_saturation=LogisticSaturationTransformer(mu=lam) 
    # Jackpot
    with plate("jackpot_plate", n_jackpot_categories):
        media_coef = numpyro.sample('media_coef', dist.HalfNormal(2))
        jackpot_effect = c_jackpot[jackpot_category]
        jackpot = numpyro.deterministic("jackpot",jnp.dot(jackpot_effect,jackpot_category))
        media_adstock = numpyro.deterministic("media_adstock", geometric_adstock.transform(X = spends)) # The spends should be wrapped with different jackpot categories to represent in a dim [spends, jackpot]
        media_saturated = numpyro.deterministic("media_saturated", logistic_saturation.transform(X = media_adstock))
        media_only = numpyro.deterministic("media_only",jnp.dot(media_coef, media_saturated))
        media_effect = numpyro.deterministic("media_effect",jnp.dot(jackpot_media_effect,media_only))


    # Media spend
    rev_sigma = numpyro.sample("sigma", dist.Uniform(0, 5))
    #TODO: Check how to do media saturation etc correctly. Should these be in the jackpot plate or where?
    rev_mean = numpyro.deterministic("mean", jackpot + media_effect + trend + seasonality)
    sales = numpyro.sample('sales', dist.Normal(rev_mean, rev_sigma), obs=sales)