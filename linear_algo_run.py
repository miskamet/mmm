import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as jnp
import os
import jax
from configs import config

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, HMC, Predictive

devices = config.devices

from models.linalgo_model import bayes_linreg_model

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
algo = NUTS(bayes_linreg_model)
mcmc = MCMC(algo, num_samples=700, num_warmup=200,num_chains=8,chain_method='parallel')
rng_key = jax.random.PRNGKey(0)
print("")
print("Starting the MCMC")
print("-----------------")
mcmc.run(jax.random.PRNGKey(0),jackpot_category=jnp.array(df_train['jackpot_size']),spends=jnp.array(df_train['media_cost']),sales=jnp.array(df_train['sales']),dayofyear=jnp.array(df_train['dayofyear']),index=jnp.array(df_train.index))
linreg_sample=mcmc.get_samples()
mcmc.print_summary(exclude_deterministic=True)
# Do some forecasting
sales_test = df_test["sales"].values
#TODO: Check again the predictions -dict mean and sales, what is the shapes of those?
predictive = Predictive(bayes_linreg_model, linreg_sample)
predictions = predictive(jax.random.PRNGKey(1),jackpot_category=jnp.array(df_test['jackpot_size']),spends=jnp.array(df_test['media_cost']),sales=jnp.array(df_test['sales']),dayofyear=jnp.array(df_test['dayofyear']),index=jnp.array(df_test.index))

y_pred = jnp.mean(predictions["sales"], axis=0)

print("mean:" +str(jnp.mean(df_test['media_cost'].values)/1000))
print("std:" +str(jnp.std(df_test['media_cost'].values)/1000))


absolute_errors = jnp.abs(y_pred - sales_test)
denominator = y_pred + sales_test  # Avoid division by zero
MAPE = jnp.mean(absolute_errors / denominator) * 100
msqrt = jnp.sqrt(jnp.mean((y_pred - sales_test) ** 2))
print("MAPE: {:.8f}, rmse: {:.8f}".format(MAPE, msqrt))

# Plot some results 
plt.figure(figsize=(18, 10))
plt.plot(df.index[::3], df["sales"][::3], label = 'true sales')
t_future = df_test.index
hpd_low, hpd_high = hpdi(predictions["sales"], prob=0.99, axis=0)
plt.axvline(x=df_train.index[-1], color='red', linestyle='--', label='Train/Test Split')
plt.plot(t_future[::3], y_pred[::3], lw=2, label = 'linear prediction')
plt.fill_between(t_future, hpd_low, hpd_high, alpha=0.3)
plt.legend(loc='upper left')
plt.ylabel('sales')
plt.xlabel('date')
plt.title("Forecasting Sales from marketing efforts (90% HPDI)")
plt.savefig('results/linalgo_prediction.png')

# Plot posterior distributions for sales data, jackpot data and media cost coefficient
coef_spends = linreg_sample["media_coef"]
c_jackpot = linreg_sample["c_jackpot"]
plt.figure(figsize=(10, 6))
fig = sns.histplot(data=coef_spends, stat="density", binwidth=1e-4)
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


# Plot prior vs posterior plots in the same histogram
fig, ax = plt.subplots()
coef_prior= np.random.normal(jnp.mean(jnp.array(df_test['media_cost'].values)/1000), jnp.std(jnp.array(df_test['media_cost'].values)/500), 200_000)
sns.histplot(data=coef_prior, stat="density",color='blue', label='Prior', alpha=0.6,binwidth=1e-2)
sns.histplot(data=coef_spends, stat="density",color='red', label = 'Posterior', alpha=0.6, binwidth=1e-4)
ax.axvline(jnp.mean(coef_spends), color='black', linestyle='--', label=f'posterior mean: {jnp.mean(coef_spends):.4f}', linewidth=0.5)
ax.axvline(jnp.mean(coef_prior), color='black', linestyle='--', label=f'prior mean: {jnp.mean(coef_prior):.4f}', linewidth = 0.5)
ax.legend(loc='best')
ax.set(title = 'prior vs posterior distributions')
plt.savefig("results/prior_vs_posterior_of_media_coef.png")
