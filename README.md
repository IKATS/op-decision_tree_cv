# Operator op-decision_tree_cv

## Decision Tree fit

This IKATS operator implements the cross-validation procedure to find the best combination of parameters, giving a more
accurate model, here the model is a decision tree.

It's using the function GridSearchCV from the popular machine learning Python library `scikit-learn` to automatically
evaluate every combination of parameters given the values that the user entered.

## Principle of the K-fold cross-validation

The K-fold cross-validation works as follow:

- the original training set is split into _K_ parts of equal length
- K-1 parts of the set are used to train the model
- the remaining part is used as a test set
- once each of the _K_ parts have been used as a test set, we take the mean of the test sets accuracies and assign it as the score of the particular parameters combination that we used

Once we have tested all the combination we wanted to, we can pick the one with the highest score and name it our
best model. Because we have iterated over several tests sets, we should be safe from overfitting on the training set.

![cv](https://static.oschina.net/uploads/img/201609/26155106_OfXx.png)

Figure 1: Overview of the cross validation procedure

### Input and parameters

This operator only takes one input:

- **Population**: of the functional type **table** (Ex: `test` output from [TrainTestSplit](https://ikats.org/doc/operators/trainTestSplit.html))

It also takes 5 inputs from the user:

- **Target**: the name of the variable we want to predict in the input table
- **ID**: the name of the rows (or table key) of the input table
- **Folds**: the number of folds, the _K_ in K-fold cross-validation, it needs to be higher than 2 and lower than the number of observations from the input table
- **Depth Parameters**: a list of values to be tested for the maximum depth of the decision tree. You can either pass a list of integers separated by a ";" or a range of values in a _pythonic_ way, for example _range(3)_ or _range(2,8)_.
- **Balance Parameters**: a list of values to be tested for the balancing parameter. If _True_ we apply a weight on our observations so thath they all matter equally, if _False_ we don't apply any weight. Like before you can test the two values if you separate them by a ";"

### Outputs

The operator has 5 outputs:

- **TDT**: a special format used by TDT viztool to show details about the built model
- **Model**: a binary dump of the best model found by the procedure, to be used by the Decision Tree Predict operator
- **Dot**: the visualisation of the best found decision tree in the GraphViz format
- **Params**: a JSON with the best combination of parameters found by the procedure
- **Result**: a **table** summarising the cross validation procedure, with informations about every combination of parameters that were tested
  - **rank**: the rank of the parameter combination according to the mean of the test accuracies of the K-fold
    cross-validation, the table is sorted according this column
  - **max_depth** & **balancing**: the values of the parameters tested on each run
  - **mean_score**: the mean of the test accuracies of the K-fold cross-validation
  - **std_score**: the standard deviation of the test accuracies of the K-fold cross-validation, we need to be sure that
    it is reasonable
