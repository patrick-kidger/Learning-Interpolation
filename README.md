# Camassa-Holm Project

A project seeking to investigate the possibility of using machine learning to perform better interpolation of solutions to PDEs. The essential idea is that solutions to a particular PDE may tend to take a particular form &ndash; knowing this, can we do better than simple polynomial interpolation?

Our PDE of choice for this project is the [Camassa&ndash;Holm equation](https://en.wikipedia.org/wiki/Camassa-Holm_equation), hence the name of the project!

### Installation

Assumptions:
* You are running Ubuntu with a desktop (e.g. [xfce](https://xfce.org/)) installed. Other Linux distros will probably work but haven't been tested. Things will probably mostly work on Windows; the `conda activate camassaholm` command below will probably have to be replaced with just `activate camassaholm`, and the 
* You already have [conda](https://conda.io/miniconda.html) installed.
* You already have [git](https://git-scm.com/) installed.
* Both git and conda are on the PATH.

First we set up the virtual environment:

```
conda create -n camassaholm
conda activate camassaholm
conda install -c conda-forge tensorflow scikit-learn jupyterlab fenics
```

(Note: It's important to install both `jupyterlab` and `fenics` in the same line; things won't work otherwise.)

Now we install [tools](https://github.com/patrick-kidger/tools). Whilst in a directory on your PYTHONPATH, run:

```
git clone https://github.com/patrick-kidger/tools.git
```

Install the repo itself (anywhere you like):

```
git clone https://github.com/patrick-kidger/camassaholmproject.git
```

Navigate to the repo and run Jupyter to start playing around:

```
cd camassaholmproject
jupyter lab
```