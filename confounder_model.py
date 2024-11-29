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
devices = config.devices


causal_graph = """
graph [
    directed 1

    node [
    id seasonality
    label "seasonality
    ]
    node [
    id trend
    label "trend"
    ]
    node [
    id jackpot_size
    label "jackpot size"
    ]
    node [
    id media_cost
    label "media cost"
    ]
    node [
    id sales
    label "sales"
    ]
    edge [
    source jackpot size
    target media cost
    ]
    edge [
    source media cost
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
    source jackpot size
    target sales
    ]
]
"""
