# README #

Splash is a header-only parallel framework that enables fast construction of MI-based gene regulatory networks (GRN). 
Splash is implemented in C++ and is parallelized for multi-core and MPI multi-node environments. 
It is designed to reconstruct networks in unsupervised and semi-supervised manners.

Splash includes the following functionalities:
  - Parallel IO to read data using exp and HDF5 formats
  - Computation kernels for computing Pearson correlation metrics.
  - Computation kernels to compute MI using Adaptive Paritioning and B-spline methods.
  - Computation kernels to compute Stouffer and CLR transformation of correlation measures.
  - ... and many more.

## Installation
At the moment the software is software is released as source code, so it is necessary to compile on your own system.

### Pre-requisites
Below are the prerequisites.  The example code are for *ubuntu* and *debian* distributions

- A modern c++ compiler with c++11 support. Supports Gnu g++, clang++ and Intel icpc.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`sudo apt install build-essential`

- MPI, for example openmpi or mvapich2

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`sudo apt install openmpi`

- cmake, and optionally cmake GUI

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`sudo apt install cmake cmake-curses-gui`

- HDF5

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`sudo apt install hdf5-helpers hdf5-tools libhdf5-dev`

## Usage

Please refere to [MCPNet](https://github.com/AluruLab/MCPNet) repositories for examples of how Splash can 
be used to construct 
