# Interative Closest Point (ICP)

## Introduction

This package is a implementation of the Iterative Closest Point (ICP) algorithm to match point clouds in Tensorflow.
It determines the translation and scale parameter between the datasets along a specified set of axes (see [Zinßer et al. 
Point Set Registration with Integrated Scale Estimation](http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2005/Zinsser05-PSR.pdf)).
It can handle datasets of an arbitrary number of dimension and runs on multiple CPUs and on GPU.


## Methods of installation

To install the repository from scratch, clone it to your computer using

```
git clone https://github.com/VincentStimper/.git
```

then go to the folder and install using `pip`

```
pip install --upgrade .
```

or use python directly in the main folder

```
python setup.py install
```



Update to the latest version directly from source

```
pip install --upgrade git+https://github.com/VincentStimper/fuller.git
```