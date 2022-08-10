
# Cross validation class

create and call cross validation for a model.


## Installation

```bash
  pip install -r requirements.txt
```
## Run

To deploy this project run

```bash
  python main.py
```


## Usage/Examples
in main.py file we should import data and create an instance of model.
then specify the cross validation configuration:

note:
every parameter start with `cv_` become a parameter for crossvalidation method and it not limited to the below parametrs.
every parameter start with `split_` become a parameter for train_test_split method and it not limited to the below parametrs.

#### configuration

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `cv_model` | `string` | cross validation model to be used e.g. KFold, TimeSeriesSplit, ShuffleSplit ... |
| `cv_n_splits`|  `int` | number of splits for cross validation|
| `cv_test_size`|  `float` | test size for cross validation|
| `cv_random_state`|  `int` | random state for cross validation|
| `cv_shuffle`|  `boolean` | shuffle for cross validation|
| `split_test_size`|   `float` | test size for train_test_split|
| `split_random_state`|  `int` | random state for train_test_split|
| `split_shuffle`|  `boolean` | shuffle for train_test_split|
| `scoring`|   `list` | scoring for cross validation e.g. ['neg_mean_squared_error', 'neg_mean_absolute_error']|
| `n_jobs`|  `int` | number of jobs for cross validation|
| `return_train_score`|  `boolean` | return train score for cross validation|
| `return_estimator`|  `boolean` | return estimator for cross validation|
| `metrics`|  `list` | metrics for cross validation regarding to scoring list names and order in list e.g. ['mean_squared_error', 'mean_absolute_error']|


``` python
    # instantiate cross validation class with model and configuration
    cv = CrossValidation(model, configuration)

    # call cross validation with features and targets
    results = cv(x, y)

    # print results
    print(results)
```
