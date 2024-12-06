import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as jnp
import os
import jax
from configs import config
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, HMC, Predictive

devices = config.devices

from models.models import bayes_linreg_model, bayes_confounder_model

plt.style.use('bmh')

df = pd.read_csv('data/data.csv', sep=';', index_col=0)
df = df[['date','year','month','week','day_of_week','dayofyear','media_cost','jackpot_size','sales']]
modeldf = df[['date','dayofyear','jackpot_size','media_cost','sales']]
print("data head")
print(df.head())
# split df to train and test
split = int(np.floor(0.8*len(df)))
df_train, df_test = df[:split], df[split:]
spends=jnp.ones(14)
jackpot_category=jnp.arange(1,15)
sales=jnp.ones(14)
dayofyear=jnp.arange(1,15)
index=jnp.arange(1,15)

numpyro.render_model(bayes_linreg_model,model_args=(jackpot_category,spends,sales,dayofyear,index), filename='models/linear_model.png', render_params=True)

# start the linear algo run
numpyro.set_host_device_count(devices)
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={devices}'
algo = NUTS(bayes_linreg_model, step_size=0.002, target_accept_prob=0.9, adapt_step_size=True)
mcmc = MCMC(algo, num_samples=700, num_warmup=300,num_chains=8,chain_method='parallel')
rng_key = jax.random.PRNGKey(0)
print("")
print("Starting the MCMC")
print("-----------------")
model_arguments = {"jackpot_category":jnp.array(df['jackpot_size']),
                   "spends":jnp.array(df['media_cost']),
                   "sales":jnp.array(df['sales']),
                   "dayofyear":jnp.array(df['dayofyear']),
                   "index":jnp.array(df.index)}
#Scale the arguments 

media_scaler = MaxAbsScaler()
sales_scaler = MaxAbsScaler()
dayofyear_scaler = MaxAbsScaler()



media_scaled = media_scaler.fit_transform(model_arguments["spends"].reshape(-1, 1))
sales_scaled = sales_scaler.fit_transform(model_arguments["sales"].reshape(-1, 1))
dayofyear_scaled = dayofyear_scaler.fit_transform(model_arguments["dayofyear"].reshape(-1, 1))


model_args = {"jackpot_category":jnp.array(df['jackpot_size'].values),
                   "spends":media_scaled,
                   "sales":sales_scaled,
                   "dayofyear":dayofyear_scaled,
                   "index":jnp.array(df.index)}

#mcmc.run(jax.random.PRNGKey(0),jackpot_category=jnp.array(df['jackpot_size']),spends=jnp.array(df['media_cost']),sales=jnp.array(df['sales']),dayofyear=jnp.array(df['dayofyear']),index=jnp.array(df.index))

mcmc.run(rng_key, **model_args)
linreg_sample=mcmc.get_samples()
mcmc.print_summary(exclude_deterministic=True)
# Do some forecasting

y_fit = jnp.mean(linreg_sample["mean"], axis=0).reshape(-1,)

MAPE = mean_absolute_percentage_error(sales_scaled, y_fit)
msqrt = root_mean_squared_error(sales_scaled, y_fit)
print("")
print("MAPE: {:.8f}, rmse: {:.8f}".format(MAPE, msqrt))
date_steps = 21
# Plot some results 
plt.figure(figsize=(18, 10))
plt.plot(df["date"][::date_steps],sales_scaled[::date_steps], label = 'true sales')
hpd_low, hpd_high = hpdi(linreg_sample["mean"].reshape(5600,1461), prob=0.80, axis=0) # Change 5600 if you change sample drawn from mcmc
plt.plot(df["date"][::date_steps], y_fit[::date_steps], label = 'predictions')
plt.fill_between(df["date"][::date_steps],hpd_low[::date_steps], hpd_high[::date_steps], alpha=0.3)
plt.legend(loc='upper left')
plt.ylabel('sales')
plt.xlabel('date')
plt.xticks(rotation=45, ha="right")
plt.title("Forecasting Sales from marketing efforts (90% HPDI)")
plt.savefig('results/linalgo_prediction.png')

# Plot posterior distributions for sales data, jackpot data and media cost coefficient
coef_spends = linreg_sample["media_coef"]
c_jackpot = linreg_sample["c_jackpot"]
plt.figure(figsize=(10, 6))
fig = sns.histplot(data=coef_spends, stat="density", binwidth=0.6)
fig.axvline(jnp.mean(coef_spends), color='black', linestyle='--', label=f'Mean: {jnp.mean(coef_spends):.4f}')
plt.legend()
fig.set_ylabel('intensity')
fig.set_xlabel('coefficient value')
fig.set_title('Posterior distribution of media coefficient')
plt.savefig("results/posterior_of_media_coef.png")

# Convert the NumPy array to a long-format DataFrame suitable for seaborn
n_samples, n_categories = c_jackpot.shape
jackpot_categories = np.repeat(np.arange(1, n_categories + 1), n_samples)  # Jackpot levels 1 to 15
samples = c_jackpot.flatten()
df_jackpot = pd.DataFrame({'Jackpot Level': jackpot_categories, 'Coefficient Value': samples})

plt.figure(figsize=(10, 6))  # Adjust figure size as needed
sns.boxplot(x='Jackpot Level', y='Coefficient Value', data=df_jackpot)
plt.title('Distribution of Jackpot Level Coefficients')
plt.xlabel('Jackpot Level')
plt.ylabel('Coefficient Value')

plt.savefig("results/posterior_of_jackpot_coef.png")

# Plot saturation curve

saturations = linreg_sample["media_saturated"]



# Changed the coef to halfnormal, rest is commented away
# Plot prior vs posterior plots in the same histogram
# fig, ax = plt.subplots()
# coef_prior= np.random.normal(jnp.mean(jnp.array(df_test['media_cost'].values)/1000), jnp.std(jnp.array(df_test['media_cost'].values)/500), 200_000)
# sns.histplot(data=coef_prior, stat="density",color='blue', label='Prior', alpha=0.6,binwidth=1e-2)
# sns.histplot(data=coef_spends, stat="density",color='red', label = 'Posterior', alpha=0.6, binwidth=1e-4)
# ax.axvline(jnp.mean(coef_spends), color='black', linestyle='--', label=f'posterior mean: {jnp.mean(coef_spends):.4f}', linewidth=0.5)
# ax.axvline(jnp.mean(coef_prior), color='black', linestyle='--', label=f'prior mean: {jnp.mean(coef_prior):.4f}', linewidth = 0.5)
# ax.legend(loc='best')
# ax.set(title = 'prior vs posterior distributions')
# plt.savefig("results/prior_vs_posterior_of_media_coef.png")
# 
#Render confounder model
spends=jnp.ones(14)
jackpot_category=jnp.arange(1,15)
sales=jnp.ones(14)
dayofyear=jnp.arange(1,15)
index=jnp.arange(1,15)

numpyro.render_model(bayes_confounder_model,model_args=(jackpot_category,spends,sales,dayofyear,index), filename='models/confounding.png', render_params=True)

#Run mcmc for confounder using the same data as before
confounder_algo = NUTS(bayes_confounder_model, step_size=0.002, target_accept_prob=0.9, adapt_step_size=True)
mcmc = MCMC(confounder_algo, num_samples=700, num_warmup=300,num_chains=8,chain_method='parallel')

mcmc.run(rng_key, **model_args)
linreg_sample=mcmc.get_samples()
mcmc.print_summary(exclude_deterministic=True)