# dev_project

## Exercise 1

### Description

In this task we have to evaluate various feature selection methods on synthetics data sets with known important and redundant attributes

### Plan

- Generate synthetic data using `make_classification()` from sklearn
- Add irrelevant features, redundant features, correlated features, noisy features etc.
- create description of data generation process
- write a function to do it, at the beginning do it in the notebook
- select at least 3 feature selection methods (at least one filter and wrapper)
    - lasso
    - bic
    - mutual information
    - correlation
- apply them to generated data
- use a measure to of success
- build model SVM and Random Forest on
    - all features
    - set obtained from selection methods
    - only relevant features
    - features from pca, mds, tsne
- compare them using accuracy, precision, recall, F1 score

## Exercise 2

### Description

### Plan
