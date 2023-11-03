# N-body simulation

N-Body simulation consisting in simulating the evolution of a dynamic system of N bodies considering their mass, positions and initial speed. 

Multiple algorithms has been designed for N-Body simulations. In this project, we focus on two of them:
  1. Brute-force is a naive algorithm that consists, for each time step, in computing the gravitational force between each pair of particle.
  2. Barnes-Hut is another algorithm with a lower complexity. Particles are stored in a tree based on their position and groups of particles can be viewed as one big particle in order to reduce the computation cost.

We present parallelized versions of these two algorithms:

  1. `cuda/` : GPU versions for the brute force algorithm. Different versions.
  2. `mpi/`: Brute force and barnes hut algorithm using MPI for load distribution.
  3. `openMP/`: Brute force and barnes hut versions parallelized using openMP.
  4. `mpi_cuda/`: Using MPI for distributing loads and force computation over GPU.
  5. `mpi_openMP`:Using MPI for distributing loads and force computation using openMP.
  6. `sequential/`: Non parallel versions.

The `test/` folder includes the `benchmark.py` to run the different tests (executables need to be complied beforehand on their respective folders). `test/bench_results` have the results of such tests, which have been done using the
lab machines at room 3a401-* and b313-* for CPU and GPU respectively.

The different plots can be visualized in the `test/plots.ipynb` notebook.
