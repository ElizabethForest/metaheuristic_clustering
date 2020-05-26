# Metaheuristic Clustering

As the name suggests this is a repository for metaheuristic clustering algorithms, implemented in Python 3, that I could not find implemented elsewhere.

Implementations are designed to work with the sklearn or pyclustering implementation style.

Currently the algorithms implemented are:
- Artifical Bee Colony (ABC)
    - Karaboga and C. Ozturk, "A novel clustering approach: Artificial Bee Colony (ABC) algorithm," Applied soft computing
- Shuffled Frog Leaping Algorithm (SFLA)
    - M. Fathian, B. Amiri, and A. Maroosi, "Application of honey-bee mating optimization algorithm on clustering," Applied Mathematics and Computation, vol. 190, no. 2, pp. 1502-1513, 2007 2007
    
## Dependencies
[Numpy](https://numpy.org/)

[PyClustering](https://github.com/annoviko/pyclustering/) 

[scikit-learn](https://scikit-learn.org/stable/) - only needed for interop with scikit-learn