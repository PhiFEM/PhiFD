# $\varphi$-FD : A well-conditioned finite difference method inspired by $\varphi$-FEM for general geometries on elliptic PDEs

This repository contains the code used in the study "$\varphi$-FD : A well-conditioned finite difference method inspired by $\varphi$-FEM for general geometries on elliptic PDEs" Michel Duprez, Vanessa Lleras, Alexei Lozinski, Vincent Vigon, Killian Vuillemot ([preprint](https://hal.science/hal-04731164)).

## This repository is for reproducibility purposes only

It is "frozen in time" and not maintained.
To use our latest $\varphi$-FEM code please refer to the [phiFEM repository](https://github.com/PhiFEM/Poisson-Dirichlet-fenicsx).

## Usage

### Prerequisites

- [Git](https://git-scm.com/),
- [Docker](https://www.docker.com/)/[podman](https://podman.io/).

The image is based on the legacy FEniCS image: quay.io/fenicsproject/stable:latest and the [`seaborn`](https://seaborn.pydata.org/) python library.

### Install the image and launch the container

1) Clone this repository in a dedicated directory:
   
   ```bash
   mkdir phifd/
   git clone https://github.com/PhiFEM/publication_PhiFD.git phifd
   ```

2) Download the images from the docker.io registry, in the main directory:
   
   ```bash
   export CONTAINER_ENGINE=docker
   cd phifd
   sudo -E bash pull-image.sh
   ```

3) Launch the container:

   ```bash
   sudo -E bash run-image.sh
   ```

### Example of usage

From the main directory `phifd`, launch the $\varphi$-FD example:

```bash
python3 phiFD_2D.py
```

## Issues and support

Please use the issue tracker to report any issues.

## Authors (alphabetical)

[Michel Duprez](https://michelduprez.fr/), Inria Nancy Grand-Est  
[Vanessa Lleras](https://vanessalleras.wixsite.com/lleras), Université de Montpellier  
[Alexei Lozinski](https://orcid.org/0000-0003-0745-0365), Université de Franche-Comté  
[Vincent Vignon](https://irma.math.unistra.fr/~vigon/), Université de Strasbourg  
[Killian Vuillemot](https://kvuillemot.github.io/), Université de Montpellier