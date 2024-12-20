# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer # Allows us to work with categorical data
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from lightgbm import plot_metric
from lightgbm import early_stopping, log_evaluation # Maybe i dont need
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_split, load_transform
from ps3.preprocessing import Winsorizer

# %%
# load data
df = load_transform()
df.head()

# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
# TODO: Why do you think, we divide by exposure here to arrive at our outcome variable? # Done
"""
The exposure is the amount of time the policyholder is covered by the insurance policy 
i.e. the risk (0.5 exposure would mean 6 months). The PurePremium is the expected claim
normalized by the exposure. Risk is not constant across all policyholders, so dividing by 
exposore allows us to compare the expected claim amount across different policyholders.
"""


# TODO: use your create_sample_split function here # Done
df = create_sample_split(df = df, id_column = "IDpol", training_frac = 0.8)
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test") # Identifies the train and test set in the main df
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy() # Creates a copy of the train and test set. This is a deep copy.

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals) # An instance of the Categorizer class is created.

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train]) # Fit_transform is not specified in the documentation, but it can be used because we call upon the class.
X_test_t = glm_categorizer.transform(df[predictors].iloc[test]) 
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5) # Tweedie distribution with power 1.5. So a mix of Poisson and Gamma distribution.
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True) 
# Fit it to the data using tweedie distribution function as a link function.
# l1_ratio is the mixing parameter for the penalty term. 1 is Lasso, 0 is Ridge.
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t) # Not sure if this alters t_glm1 so I redifined it another model above to use in the pipeline below.


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)

"""
Easier to interprest loss values relative to each other. The loss on the test set is higher than the training set, which is expected. 
"""

# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps: 
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer. 
#    Choose knots="quantile" for the SplineTransformer and make sure, we 
#    are only including one intercept in the final GLM. 
# 2. Put the transforms together into a ColumnTransformer. 
#    Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Let's put together a pipeline
numeric_cols = ["BonusMalus", "Density"]
numeric_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()), # Standardizes the features by removing the mean and scaling to unit variance.
    ('spline transformer', SplineTransformer(knots="quantile")) # SplineTransformer is a class that allows us to transform the data into splines according to the knots specified.
])
preprocessor = ColumnTransformer( # Instance of the class ColumnTransformer from the sklearn package.
    transformers=[
        # TODO: Add numeric transforms here # Done
        ("num", numeric_pipeline, numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)
# In a pipeline, you have lists of tuples where the first element is the name of the step and the second element is the transformer object.

t_glm2 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True) # Redefining the model to use in the pipeline below. 

preprocessor.set_output(transform = "pandas")
model_pipeline = Pipeline(steps=[
    # TODO: Define pipeline steps here # Done
    ('preprocessor', preprocessor),
    ('model', t_glm2)
])   

# let's have a look at the pipeline
model_pipeline

# %%

# let's check that the transforms worked
X = model_pipeline[:-1].fit_transform(df_train) # Need to attribute it to some sort of variable 
# Can call pipeline steps with bracket indexing.

#print(X)

model_pipeline.fit(df_train, y_train_t, model__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.


model_pipeline = Pipeline(steps=[
    # TODO: Define pipeline steps here # Done
    ('preprocessor', preprocessor),
    ('model', LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5))
])

model_pipeline.fit(X_train_t, y_train_t, model__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

"""
Training loss is a lot lower than testing loss. Might want to treat it so that the loss difference is closer. Don't want to overfit.
"""

# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param. 

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators

param_grid = {
    'model__learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1],
    'model__n_estimators': [50, 100, 200]
}
# What are good values of leaning rate? It's an art. Lower learning rates are better for smaller datasets, but take longer to train.

cv = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2, # Test each combination of each parameter value. Evaluetes the fit using the cross-validation set.
    n_jobs=-1
)

cv.fit(X_train_t, y_train_t, model__sample_weight=w_train_t)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm_cv:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm_cv:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)


# %%
# Let's compare the sorting of the pure premium predictions


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()

"""
(0,0) is the safest driver in terms of pure premium. And (1,1) is the riskiest driver.
Random baseline means the risk has nothing to do with the policy premium.
LGBM does better than the baseline and is the most pronounced curve compared to GLM.
The Oracle is the benchmark, and it means that almost all the claims are driven by a very very small fraction of the policyholders.
"""

#%%
############################################################## PS 4 ################################################################
# %%
###### Qustion 1: Monotonicity Constraints ######
# TODO: Create a plot of the average claims per BonusMalus group, make sure to weigh them by exposure. # Done
# What will/could happen if we do not include a monotonicity constraint?  # Done

plt.figure(figsize=(10, 6))
df.groupby("BonusMalus")["PurePremium"].mean().plot() # Plotting average pure pqremium per BonusMalus. Using pure premium because it is normalised by exposure.
plt.xlabel("BonusMalus")
plt.ylabel("Average Pure Premium")
plt.title("Average Pure Premium per BonusMalus")
plt.show()

"""
Pure premium is affected by BonusMalus, but not in a strictly increasing manner. 
If we do not include a monotonicity constraint, the model could predict a decrease pure premium for a higher BonusMalus.
This is intuitively unrealistic despite it existing in the data.
"""

# TODO: Create a new model pipeline or estimator called constrained_lgbm # Done
# Introduce an increasing monotonicity constrained for BonusMalus. # Done
# Note: We have to provide a list of the same length as our features with 0s everywhere except for BonusMalus where we put a 1. # Done

const_predictors = preprocessor.get_feature_names_out() # Get the feature names from the preprocessor.
#print(len(const_predictors)) # 64

monotonicity_constraint = [0] * len(const_predictors)
bonus_malus_indices = [i for i, name in enumerate(const_predictors) if "BonusMalus" in name]
for i in bonus_malus_indices:
    monotonicity_constraint[i] = 1

constrained_lgbm =  LGBMRegressor(
    objective = 'tweedie', 
    tweedie_variance_power = 1.5, 
    monotone_constraints = monotonicity_constraint,
    monotone_constraints_method = 'basic'
)

constrained_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', constrained_lgbm)
])

constrained_pipeline

# TODO: Cross-validate and predict using the best estimator. Save the predictions in the column pp_t_lgbm_constrained. # Done

cv_cons = GridSearchCV(
    estimator=constrained_pipeline,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2,
    n_jobs=-1
)

cv_cons.fit(X_train_t, y_train_t, model__sample_weight=w_train_t)

df_test["pp_t_lgbm_constrained"] = cv_cons.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm_constrained"] = cv_cons.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm_constrained"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm_constrained"]),
    )
)


# %%
###### Qustion 2: Learning Curve ######
# TODO: Re-fit the best constrained lgbm estimator from the cross-validation and
# provide the tuples of the test and train dataset to the estimator via eval_set

best_constrained_pipeline = cv_cons.best_estimator_
best_constrained_pipeline

best_constrained_lgbm = best_constrained_pipeline.named_steps['model']

# Refit the model with eval_set
best_constrained_pipeline.fit(
    X_train_t, 
    y_train_t,
    model__sample_weight=w_train_t,
    model__eval_set=[(X_train_t, y_train_t), (X_test_t, y_test_t)],
    model__eval_metric='tweedie'
)
"""
The block of code above does not work and throws an error about training data not being in the same shape as the test data.
i don't know why this is the case. I tried to fix it. Does not work. So will wait for solutions.
"""
#%%
# Predict on train and test sets
df_train["pp_t_lgbm_constrained_refit"] = best_constrained_pipeline.predict(X_train_t)
df_test["pp_t_lgbm_constrained_refit"] = best_constrained_pipeline.predict(X_test_t)

print(
    "Refit training loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(
            y_train_t, df_train["pp_t_lgbm_constrained_refit"], sample_weight=w_train_t
        ) / np.sum(w_train_t)
    )
)

print(
    "Refit testing loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(
            y_test_t, df_test["pp_t_lgbm_constrained_refit"], sample_weight=w_test_t
        ) / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm_constrained_refit"]),
    )
)

#%%
# TODO: Plot the learning curve by running lgb.plot_metric on the estimator 
# (either the estimator directly or as last step of the pipeline)

plot_metric(
    booster = best_constrained_lgbm,
    metric = 'tweedie',
    title = 'Learning Curve',
    figsize = (10, 6),
    xlabel = 'Iterations',
    ylabel = 'Tweedie Deviance'
)


# What do you notice, is the estimator tuned optimally?
# %%
