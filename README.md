# Metawrappers
A Python library of metaheuristic-based feature selection wrappers. Compatible with `scikit-learn`.

**Warning**: This was created solely for reasearch / learning purposes.
In my tests everything worked ok, but please, be careful if you plan on using this for anything serious.

Also, I probably won't actively develop this anymore.

## About
The library implements six feature selection wrappers based on different metaheuristic techniques:
- Hill Climbing - `HCSelector`
- Simulated Annealing - `SASelector`
- Tabu Search - `TSSelector`
- Particle Swarm Optimization - `PSOSelector`
- Ant Colony Optimization - `ACOSelector`
- Genetic Algorithm - `GASelector`

There's also `RandomSelector` which literally just selects a random subset of features, I used it as a baseline for some tests.

Additionaly, there's `experiments.Comparator` class which can be used for easily comparing performance of the selectors (or any `scikit-learn` estimators for that matter).

## Usage

### Installation
```shell
pip install git+https://github.com/bcyran/metawrappers.git
```

### Selection

#### Basic example
This is how a basic, self-contained usage example of `ACOSelector` could look:
```python
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from metawrappers import ACOSelector

X, y = load_wine(return_X_y=True)
selector = ACOSelector(SVC())  # create the selector using SVC classifier
selector.fit(X, y)  # Perform the metaheuristic selection
print(selector.get_support())  # Print out the feature mask
X = selector.transform(X)  # Transfrom the samples vector
```

#### Dimensionality reduction
Be default, the selectors will try to maximize the classification accuracy without taking the number of selected features into account.
If you would like to minimize the number of attributes, you can use the `feature_num_penalty` constructor param, e.g.:
```python
selector = ACOSelector(SVC(), feature_num_penalty=0.1)
```
This works for all algorithms.

#### Parameters
Each of the selectors' constructor takes a number of metaheuristic-specific parameters which can dramatically change the results.
The defaults are what seemed sane to me, but for basically any purpose you'll probably want to play around with those.
For the parameters' documentation please refer to the selector classes docstrings.

#### Limiting run-time
Default stopping condition is usually based on iteration count.
The time it takes to execute specific number of iterations is hugely dependent on used dataset, in some cases it could take a really long time
It's often a good idea to set a specific run duration, e.g. 5 seconds:
```python
selector = ACOSelector(SVC(), run_time=5000)
```
This supersedes all other stopping conditions based on iterations, no matter if it's a default or explicit value.

#### Some shortcuts
For convenience, the selectors implement also `predict()`, `score()` and some other methods which are delegated to the underlying estimator, e.g.:
```python
selector = ACOSelector(SVC(), run_time=5000)
selector.fit(X, y)
predicted = selector.predict(X)  # We don't have to manually transform or call the classifier
```
This automatically transforms the samples vector given to `predict` and uses the underlying `SVC` instance to obtain the predictions.

### Experiments
If you want to compare the results of different algorithms, `Comparator` class can be useful.
For instance, to compare the results which ACO, PSO and SA can obtain in one second, you could run the following:
```python
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from metawrappers import ACOSelector, PSOSelector, SASelector
from metawrappers.experiments import Comparator

X, y = load_wine(return_X_y=True)
selectors = {
    "None": SVC(),
    "ACO": ACOSelector(SVC(), run_time=1000),
    "PSO": PSOSelector(SVC(), run_time=1000),
    "SA": SASelector(SVC(), run_time=1000),
}
comparator = Comparator(selectors, n_tests=20, n_splits=5)
comparator.run(X, y)  # Runs the comparison and automatically prints a summary
comparator.results  # You can also access pandas dataframe with full results
```
This will run 20 repeats of 5-fold cross validation with each of the specified estimators.
Also, a Wilcoxon signed-rank test will be performed for accuracy obtained by each selector, treating the first one (in this case just `SVC()` without any selection) as a baseline.
In the end, short summary will be printed, e.g.:
```
Comparison summary:
      #feat  avg score  max score         p  h0 rejected
None  13.00   0.679873   0.724762       NaN        False
ACO    4.35   0.909905   0.932540  0.000002         True
PSO    5.85   0.913500   0.933016  0.000002         True
SA     5.70   0.903381   0.927619  0.000002         True
```
As you can see, all selectors improved the results in a statistically significant way.
