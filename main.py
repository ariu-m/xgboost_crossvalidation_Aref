import xgboost as xgb
import pandas as pd
from cross_validation import CrossValidation

data = pd.read_csv("Caiso_DAM_Price.csv")
data2 = data[[ 'hour', 'enhanced_weekday',
       'outage_total_modified_sma', 'previous_day',
       'previous_hour', 'previous_week', 
        'Wind_Solar_Total_diff', 
       'Weather_SP15_humidity', 'FRSCE1_fuel_price', 'Solar_7DA',
       'load_gen_diff', 'target0']]

# targets
y = data2["target0"].values
data2.pop('target0')

# features
x = data2

conf = {
        'cv_model': 'KFold', # string: cross validation model to be used (KFold, TimeSeriesSplit, ShuffleSplit)
        'cv_n_splits': 5, # int: number of splits for cross validation
        # 'cv_test_size': 0.2, # float: test size for cross validation
        'cv_random_state': 42, # int: random state for cross validation
        'cv_shuffle': True, # boolean: shuffle for cross validation
        'split_test_size': 0.2, # float: test size for train_test_split
        'split_random_state': 42, # int: random state for train_test_split
        'split_shuffle': True, # boolean: shuffle for train_test_split
        'scoring': ['neg_mean_absolute_error', 'neg_mean_squared_error'], # list: scoring for cross validation
        'n_jobs': 2, # int: number of jobs for cross validation
        'return_train_score': True, # boolean: return train score for cross validation
        'return_estimator': True, # boolean: return estimator for cross validation
        'metrics': ['mean_absolute_error', 'mean_squared_error'], # list: metrics for cross validation regarding to scoring list
        }

# instantiate of model
xgbr = xgb.XGBRegressor(
    booster="gbtree",
    sample_type="uniform",
    normalize_type="tree",
    skip_drop=0,
    rate_drop=0,
    n_estimators=65,
    learning_rate=0.19,
    gamma=0,
    max_depth=5,
    min_child_weight=2.2,
    max_delta_step=0,
    reg_lambda=0,
    reg_alpha=5,
    base_score=0.5,
    eval_metric="rmse",
    objective="reg:squarederror",
    random_state=0
)

# instantiate cross validation class with model and configuration
cv = CrossValidation(xgbr, conf)

# call cross validation with features and targets
r = cv(x, y)

print(r)