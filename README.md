# 02445 Task 2: Evaluating Predictive Models

Individual assignment for the course **02445 Project in Statistical Evaluation** (DTU), Spring 2024.

The [report](report.pdf) investigates whether heart-rate summary
statistics can predict a participant's self-reported frustration level. Five
models (an artificial neural network, random forest, k-nearest neighbours,
ridge regression, and logistic regression) are compared against a mean baseline
using group-aware 8-fold cross-validation and evaluated with a Friedman test and
Bonferroni-corrected post-hoc comparisons. The conclusion is a negative result:
none of the models significantly outperform the baseline, indicating that the
heart-rate summary statistics carry little usable signal about frustration in
this setup.

## Data

The analysis uses `HR_data.csv`, the summary statistics of the heart-rate
recordings from the EmoPairCompete dataset (Das et al., 2024). This file is not
included in the repository; place it in the project root before running the
notebooks.

`final_errors.csv` holds the validation errors each model obtained through the
cross-validation described above, where only data from separate cohorts and
rounds was used to train the models. `ConsistencyErrors.pkl` is a Python
dictionary of the errors from the model-consistency evaluation.

## Code

- `Data_analysis.ipynb`: the main data analysis and model comparison for the report.
- `ConsistencyTester.ipynb`: the consistency analysis of the models.
- `BestModelFinders.py`: helper functions (hyperparameter search, model definitions) used by `Data_analysis.ipynb`.
- `ConsistencyTesterFunctions.py`: the same helpers used by `ConsistencyTester.ipynb`, with the random state removed for the consistency runs.

## Reference

Das, S. et al. (2024). *EmoPairCompete: Physiological signals dataset for emotion
and frustration assessment under team and competitive behaviours.* ICLR 2024
Workshop on Learning from Time Series for Health.
