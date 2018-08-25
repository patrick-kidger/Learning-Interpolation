# Interpolating PDE solutions using feedforward neural networks

A project seeking to investigate the possibility of using machine learning to perform better interpolation of solutions to PDEs. The essential idea is that solutions to a particular PDE may tend to take a particular form &ndash; knowing this, can we do better than simple polynomial interpolation?

Our PDE of choice for this project is the [Camassa&ndash;Holm equation](https://en.wikipedia.org/wiki/Camassa-Holm_equation).

### Installation

Assumptions:
* You are running Ubuntu with a desktop (e.g. [xfce](https://xfce.org/)) installed.
  * Other Linux distros will probably work but haven't been tested.
  * Windows will probably mostly work but also hasn't been tested. The `conda activate camassaholm` command below will probably have to be replaced with just `activate camassaholm`.
* You already have [conda](https://conda.io/miniconda.html) installed.
* You already have [git](https://git-scm.com/) installed.
* Both git and conda are on the PATH.

First we set up the virtual environment:

```
conda create -n camassaholm -c conda-forge --file requirements.txt
conda activate camassaholm
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```


Now we install [tools](https://github.com/patrick-kidger/tools), and checkout the version used for this project (which is at time of writing the most recent commit.) Whilst in a directory on your PYTHONPATH, run:

```
git clone https://github.com/patrick-kidger/tools.git
cd tools
git checkout 00e3f8009e0a0b4288812a6da78272edf6a6475e
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

### Usage

There are a few main scripts for doing the actual learning in the project/ directory; each can be run on its own. For testing their results, and testing things about the project in general, there are a few (somewhat ad-hoc) scripts in project/test/.

The base code providing the system on which all of this is built is found in the project/base/ directory. The saved models are (surprise) saved to the project/saved_models/ directory, although these have not been uploaded to GitHub!
