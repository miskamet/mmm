import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as jnp
import os
import jax
from configs import config
from dowhy import CausalModel
import numpyro
import pandas as pd



devices = config.devices


causal_graph = """
graph [
    directed 1

    node [
        id seasonality
        label "seasonality"
    ]
    node [
        id trend
        label "trend"
    ]
    node [
        id jackpot_size
        label "jackpot_size"
    ]
    node [
        id media_cost
        label "media_cost"
    ]
    node [
        id sales
        label "sales"
    ]
    edge [
        source jackpot_size
        target media_cost
    ]
    edge [
        source media_cost
        target sales
    ]
    edge [
        source trend
        target sales
    ]
    edge [
        source seasonality
        target sales
    ]
    edge [
        source jackpot_size
        target sales
    ]
]
"""
plt.style.use('bmh')

df = pd.read_csv('data/data.csv', sep=';', index_col=0)
df = df[['date','year','month','week','day_of_week','dayofyear','media_cost','jackpot_size','sales']]

modeldf = df[['date','dayofyear','jackpot_size','media_cost','sales']]

print("data head")
print(modeldf.head())
# split df to train and test
split = int(np.floor(0.8*len(modeldf)))
model_train, model_test = modeldf[:split], modeldf[split:]

causal_model = CausalModel(data=model_train,graph=causal_graph,treatment='media_cost',outcome='sales')
causal_model.view_model()
print("")
print("Backdoor path for causal graph:")
print(causal_model._graph.get_backdoor_paths(nodes1=["media_cost"], nodes2=["sales"]))
print("")
print("Confirm the D-separation for nodes media, jackpot and sales:")
print(causal_model._graph.check_valid_backdoor_set(nodes1=["media_cost"], nodes2=["sales"], nodes3=["jackpot_size"]))