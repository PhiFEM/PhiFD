# $\phi$-FD : A well-conditioned finite difference method inspired by $\phi$-FEM for general geometries on elliptic PDEs

**Authors: Michel Duprez, Vanessa Lleras, Alexei Lozinski, Vincent Vigon, and Killian Vuillemot**

This repository contains all codes to reproduce results of the paper "$\phi$-FD : A well-conditioned finite difference method inspired by $\phi$-FEM for general geometries on elliptic PDEs", in collaboration with Michel Duprez, Vanessa Lleras, Alexei Lozinski, and Vincent Vigon. 

The directory `./Convergence/` contains the Python codes to run all the methods for 2D and 3D cases, to analyze the convergence of each method.
The `./Multigrid/` contains everything to run the multigrid approach on 2D and 3D test cases. 

To execute these codes, you will need several packages : 
[*FEniCS*](https://fenicsproject.org/),
[*numpy*](https://numpy.org/doc/stable/index.html),
[*matplotlib*](https://matplotlib.org/),
[*seaborn*](https://seaborn.pydata.org/),
[*pandas*](https://pandas.pydata.org/),
[*SciPy*](https://scipy.org/).

The easiest way to perform these installations is by using Anaconda, with for example the following lines: 

```bash 
conda env create -f phiFD.yml 
conda activate phiFD
conda install -c conda-forge superlu_dist=6.2.0
pip3 install mpi4py==3.0.3 --no-binary mpi4py --user --force --no-cache-dir
pip install numpy matplotlib seaborn pandas scipy 
```
