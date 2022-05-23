# Metaheuristic Clustering

[![version status](https://img.shields.io/pypi/v/metaheuristic_clustering.svg?style=flat)](https://pypi.org/project/metaheuristic-clustering)

As the name suggests, this is a repository for metaheuristic clustering algorithms, implemented in Python 3, that I could not find implemented elsewhere.

Implementations are designed to work with or without the sklearn implementation style.

Currently the algorithms implemented are:
- Artifical Bee Colony (ABC)
    - D. Karaboga and C. Ozturk (2011). "A novel clustering approach: Artificial Bee Colony (ABC) algorithm", Applied soft computing, 11(1), 652–657, 2011.
- Shuffled Frog Leaping Algorithm (SFLA)
    - B. Amiri, M. Fathian, and A. Maroosi (2009). "Application of shuffled frog-leaping algorithm on clustering", The International Journal of Advanced Manufacturing Technology, 45(1-2), 199-209.


## Installation

metaheristic_clustering can be installed with:

```bash
pip install metaheuristic-clustering
```

Or you can [fork this repository](https://github.com/ElizabethForest/metaheuristic_clustering/fork)

    
## Dependencies
[Numpy](https://numpy.org/)

[PyClustering](https://github.com/annoviko/pyclustering/) 

[scikit-learn](https://scikit-learn.org/stable/) - only needed for interop with scikit-learn


## Contributing Guidelines

Please create a [pull request](https://github.com/ElizabethForest/metaheuristic_clustering/pulls) or an [issue](https://github.com/ElizabethForest/metaheuristic_clustering/issues) if you would like to contribute or have any bug reports, issues, or suggestions.


## Example

There is example code using the metaheuristic-clusteing library available:
- [python file](https://github.com/ElizabethForest/metaheuristic_clustering/blob/master/example.py)
- [jupyter notebook](https://github.com/ElizabethForest/metaheuristic_clustering/blob/master/example.ipynb)

Or for a breif overview see below:

### Sklearn/Object style

```python
data = X  # your data

# SFLA Clustering
from metaheuristic_clustering.sfla import SFLAClustering

sfla_model = SFLAClustering()
sfla_labels = sfla_model.fit_predict(data)

# ABC Clustering
from metaheuristic_clustering.abc import ABCClustering

abc_model = ABCClustering()
abc_labels = abc_model.fit_predict(data)
```

### Function style

```python
import metaheuristic_clustering.util as util

data = X  # your data

# SFLA Clustering
import metaheuristic_clustering.sfla as sfla

best_frog = sfla.sfla(data)
sfla_labels = util.get_labels(data, best_frog)

# ABC Clustering
import metaheuristic_clustering.abc as abc

best_bee = abc.abc(data)
abc_labels = util.get_labels(data, best_bee)
```

### Sample Results

#### ABC
![Graphs of ABC Results](https://github.com/ElizabethForest/metaheuristic_clustering/blob/master/ABC_results.png)

#### SFLA
![Graphs of SLFA Results](https://github.com/ElizabethForest/metaheuristic_clustering/blob/master/SFLA_results.png)


