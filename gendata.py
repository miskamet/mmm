import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import GeometricAdstockTransformer, LogisticSaturationTransformer, BetaHillTransformation, jackpotgenerator
from scipy.ndimage import gaussian_filter





#set up variables
seed = 69420
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# date range
min_date = pd.to_datetime("2020-01-01")
max_date = pd.to_datetime("2023-12-31") 

df = pd.DataFrame(
    data={"date": pd.date_range(start=min_date, end=max_date, freq="D")}
)
df = df.assign(
    year = lambda x: x["date"].dt.year,
    month = lambda x: x["date"].dt.month,
    week = lambda x: x["date"].dt.isocalendar().week,
    day_of_week = lambda x: x["date"].dt.isocalendar().day,
    dayofyear = lambda x: x["date"].dt.dayofyear
)
n = len(df)

# Simulate media cost feature using uniform distribution
media_cost = np.random.uniform(0,1, size=n)
df["media_cost"] = 100*np.where(media_cost >0.3, media_cost, media_cost/4)
# visualize the media cost using only sundays (does not matter cause the data is uniform)
df_subset = df[df["day_of_week"] == 7]

fig, ax = plt.subplots()
sns.lineplot(x="date", y="media_cost", data=df_subset, ax=ax)
ax.set(title="Raw data for media costs")
plt.xticks(rotation=45, ha="right")
plt.savefig('datagen_images/media_cost.png')

#initialize the adstock effects you want
alpha = 0.5
# give the length of adstock effect in days. In veikkaus, this is quite small, since there is all the time new stuff coming
l = 14
geometric_adstock_transformer = GeometricAdstockTransformer(alpha=alpha, l=l)

# add transformed variables to data

df["media_cost_adstock"] = geometric_adstock_transformer.transform(X=df["media_cost"])

# Plot and save the transformed data

fig, axes = plt.subplots(
  nrows=2,
  ncols=1,
  sharex=True,
  sharey=False,
  layout="constrained"
)

features = ["media_cost", "media_cost_adstock"]
df_subset = df[df["day_of_week"] == 7]
for i, (col, ax) in enumerate(zip(features, axes.flatten())):
    sns.lineplot(x="date", y=col, color=f"C{i}", label=col, data=df_subset, ax=ax)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

fig.suptitle("Media Cost with adstock transformation")
plt.savefig('datagen_images/media_cost_adstock_vs_without.png')

# logistic saturation transformation for the media costs

mus = [1e-2,2e-2,3e-2,4e-2]
fig, ax = plt.subplots(figsize=(7, 6))

for mu in mus:
    logistic_saturation_transformer = LogisticSaturationTransformer(mu=mu)

    df[f"media_cost_saturation_mu{mu:.2f}"] = logistic_saturation_transformer.fit_transform(df["media_cost_adstock"])

    df_subset = df[df["day_of_week"] == 7]


    sns.lineplot(
        x="media_cost_adstock",
        y=f"media_cost_saturation_mu{mu:.2f}",
        label=f"mu = {mu}",
        data=df_subset,
        ax=ax
    )
ax.legend(loc="lower right", prop={"size": 18})
ax.set_ylabel('media cost saturation')
ax.set(title=f"Saturation curve with logistic regression");

plt.savefig('datagen_images/media_cost_saturation_curve_logistic_transform.png')

# Test beta-hill transformation
K = 100
S = 6
B = 1
bh_transformer = BetaHillTransformation(K=K, S=S, beta=B)
df["media_cost_saturation_beta_hill"] = bh_transformer.transform(df["media_cost_adstock"])
# Plot the results
df_subset = df[df["day_of_week"] == 7]

fig, ax = plt.subplots(figsize=(7, 6))
sns.lineplot(
    x="media_cost_adstock",
    y=f"media_cost_saturation_beta_hill",
    label=f"K = {K}\n S = {S}\n beta = {B}",
    data=df_subset,
    ax=ax
)
ax.legend(loc="lower right", prop={"size": 18})
ax.set(title=f"Saturation curve with Beta Hill transformation");

plt.savefig('datagen_images/media_cost_saturation_curve_beta_hill_transform.png')

fig, axes = plt.subplots(
  nrows=4,
  ncols=1,
  sharex=True,
  sharey=False,
  layout="constrained"
)

features = ["media_cost", "media_cost_adstock","media_cost_saturation_mu0.02", "media_cost_saturation_beta_hill"]
for i, (col, ax) in enumerate(zip(features, axes.flatten())):
    sns.lineplot(x="date", y=col, color=f"C{i}", label=col, data=df_subset, ax=ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('')

fig.suptitle("Transformations made for media cost")
plt.xticks(rotation=45, ha="right")


plt.savefig('datagen_images/media_cost_all_transformations.png')

# Apply dim return curve effect

df["beta"] =  (np.arange(start=0.0, stop=1.0, step=1/n) + 1) ** (-1.8)
df_subset = df[df["day_of_week"] == 7]

fig, ax = plt.subplots()
sns.lineplot(x="date", y="beta", color="C3", data=df_subset, ax=ax)
ax.set(title="Diminishing return curve over time", ylabel=None)
ax.set_ylabel(r"$\beta$ value")
plt.savefig('datagen_images/diminishing_return_curve.png')

df["media_effect"] = df["beta"] * df["media_cost_saturation_mu0.02"]

features = ["media_cost", "media_cost_adstock","media_cost_saturation_mu0.02", "media_cost_saturation_beta_hill", "media_effect"]

df_subset = df[df["day_of_week"] == 7]

fig, axes = plt.subplots(
    nrows=5,
    ncols=1,
    figsize=(12, 7),
    sharex=True,
    sharey=False,
    layout="constrained"
)
for i, (col, ax) in enumerate(zip(features, axes.flatten())):
    sns.lineplot(x="date", y=col, color=f"C{i}", label=col, data=df_subset, ax=ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('')

fig.suptitle("All transformations for media costs")
plt.xticks(rotation=45, ha="right")

plt.savefig('datagen_images/media_cost_all_transformations_with_media_effect.png')

g = sns.lmplot(
    x="media_cost_adstock",
    y="media_effect",
    hue="year",
    palette="husl",
    lowess=True,
    scatter_kws={"edgecolor": "black", "alpha": 0.7},
    height=6,
    data=df
)
g.ax.set_xlabel("media cost with adstock")
g.ax.set_ylabel("media effect")
g.figure.suptitle("Media Cost Effect", y=1.02);
plt.savefig('datagen_images/media_cost_effect_yearly.png')

df.eval(expr="effect_ratio = media_effect / media_cost_adstock", inplace=True)

df["effect_ratio_smooth"] = gaussian_filter(input=df["effect_ratio"], sigma=8)
df_subset = df[df["day_of_week"] == 7]

fig, ax = plt.subplots()
sns.lineplot(x="date", y="effect_ratio", color="C10", label="effect ratio", data=df_subset, ax=ax)
sns.lineplot(x="date", y="effect_ratio_smooth", color="C15", label="effect ratio smooth", data=df_subset, ax=ax)
ax.set(title="Media Cost Effect Ratio")
plt.savefig(f'datagen_images/media_cost_effect_ratio.png')

# Simulate trend and seasonality
df["trend"] = (np.linspace(start=0.0, stop=50, num=n) + 10)**(1/3) - 1 

df["cs"] = - np.sin(1 * 2 * np.pi * df["dayofyear"] / 365.5) 
df["cc"] = np.cos(2 * 2 * np.pi * df["dayofyear"] / 365.5)
df["seasonality"] = 0.2 * (df["cs"] + df["cc"])
df_subset = df[df["day_of_week"] == 7]

fig, ax = plt.subplots()
sns.lineplot(x="date", y="trend", color="C4", label="trend", data=df_subset, ax=ax)
sns.lineplot(x="date", y="seasonality", color="C6", label="seasonality", data=df_subset, ax=ax)
ax.legend(loc="upper left")
ax.set(title="Trend and Seasonality", ylabel="")
plt.savefig(f'datagen_images/trend_seasonality.png')

#Create intercept variable and noise for the data
np.random.seed(seed)

# Generate a jackpot sizes for the model and jackpot effects regarding those.
jackpotsizes = jackpotgenerator(df["date"].tolist())
df["jackpot_size"] = jackpotsizes


# Generate a jackpot size effect for the media effect(Confounder)
# media_effect_with_jackpot = jackpot_size * media_effect/3, therefore the media will be worse when jackpot size is small
df.eval(expr="media_effect_with_jp = media_effect * jackpot_size*0.2", inplace=True)

df_subset = df[df["day_of_week"] == 7]
fig, ax = plt.subplots()
sns.lineplot(x="date", y="media_effect", color="C2", label="media effect", data=df_subset, ax=ax)
sns.lineplot(x="date", y="media_effect_with_jp", color="C8", label="media effect with jackpot size", data=df_subset, ax=ax)
ax.legend(loc="upper left")
ax.set(title="Media effect with jackpot size", ylabel="effect")
plt.savefig(f'datagen_images/mediaeffect_w_jackpot.png')

df["intercept"] = 3.0
df["trend_plus_intercept"] = df["trend"] + df["intercept"]

# the noise variance is increasing to make sure the resulting time series has constant variance
sigma_epsilon  = np.linspace(start=3e-2, stop=7e-2, num=n)
df["epsilon"] = np.random.normal(loc=0.0, scale=sigma_epsilon)
df.eval(expr="sales = intercept + trend + seasonality + media_effect_with_jp + epsilon + jackpot_size", inplace=True)

df_subset = df[df["day_of_week"] == 7]
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(15)
sns.lineplot(x="date", y="sales", color="blue", data=df_subset, ax=ax)
ax.set(title="Generated sales of the product")

plt.savefig(f'datagen_images/sales_of_product.png')

print("Data generated:")
print(df.head())
print("info about the data")
print(df.info())
df.to_csv('data/data.csv', sep=';', index=True)
print("")
print("DATA SAVED!")
print("-----------")
