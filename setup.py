from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="metaheuristic_clustering",
    version="0.0.1",
    author="Elizabeth Ditton",
    author_email="elizabeth.forest@my.jcu.edu.au",
    description="sklearn and pyclustering style implementations of SFLA and ABC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["metaheuristic_clustering"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.7',
    packages=['metaheuristic_clustering'],
    package_dir={'': 'src'},
    install_requires=['pyclustering', 'sklearn', 'numpy'],
    url="https://github.com/ElizabethForest/metaheuristic_clustering"
)
