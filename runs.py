import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as jnp
import os
import jax

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, HMC, Predictive


from models import bayes_linreg_model

df = pd.read_csv('data/data.csv', sep=';', index_col=0)
df = df[['date','year','month','week','day_of_week','dayofyear','media_cost_adstock','jackpot_size','sales']]

print("data head")
print(df.head())
# split df to train and test
split = int(np.floor(0.8*len(df)))
df_train, df_test = df[:split], df[split:]

# start the linear algo run
numpyro.set_host_device_count(4)
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
algo = NUTS(bayes_linreg_model)
mcmc = MCMC(algo, num_samples=1000, num_warmup=500,num_chains=4,chain_method='parallel')
rng_key = jax.random.PRNGKey(0)
mcmc.run(jax.random.PRNGKey(0),jackpot_category=jnp.array(df_train['jackpot_size']),spends=jnp.array(df_train['media_cost_adstock']),sales=jnp.array(df_train['sales']),dayofyear=jnp.array(df_train['dayofyear']),index=jnp.array(df_train.index))
linreg_sample=mcmc.get_samples()
mcmc.print_summary(exclude_deterministic=True)

# Do some forecasting
sales_test = df_test["sales"].values
#TODO: Check again the predictions -dict mean and sales, what is the shapes of those?
predictive = Predictive(bayes_linreg_model, linreg_sample)
predictions = predictive(jax.random.PRNGKey(1),jackpot_category=jnp.array(df_test['jackpot_size']),spends=jnp.array(df_test['media_cost_adstock']),sales=jnp.array(df_test['sales']),dayofyear=jnp.array(df_test['dayofyear']),index=jnp.array(df_test.index))
y_pred = predictions["sales"]
y_pred = y_pred[:len(sales_test)-1]

sMAPE = jnp.mean(jnp.abs(y_pred - sales_test) / (y_pred + sales_test)) * 200
msqrt = jnp.sqrt(jnp.mean((y_pred - sales_test) ** 2))
print("sMAPE: {:.2f}, rmse: {:.2f}".format(sMAPE, msqrt))

# Plot some results 
plt.figure(figsize=(8, 4))
plt.plot(df["date"], df["sales"])
t_future = df_test["date"]
hpd_low, hpd_high = hpdi(y_pred)
plt.plot(t_future, y_pred, lw=2)
plt.fill_between(t_future, hpd_low, hpd_high, alpha=0.3)
plt.title("Forecasting Sales from marketing efforts (90% HPDI)")
plt.show()