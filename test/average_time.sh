#!/bin/bash

# Number of times to run the simulation
num_runs=10

# Initialize total execution time
total_time=0

# Number of particles and simulation end time
nparticles=1000  # You can change this value
T_FINAL=2.0     # You can change this value

for ((i = 1; i <= num_runs; i++)); do
  echo "Run $i:"
  # Sequential
  #duration=$(../sequential/nbody_brute_force $nparticles $T_FINAL | grep "Simulation took" | awk '{print $3}')
  #duration=$(../sequential/nbody_barnes_hut $nparticles $T_FINAL | grep "Simulation took" | awk '{print $3}')

  #openMP
  #duration=$(../openMP/nbody_brute_force $nparticles $T_FINAL | grep "Simulation took" | awk '{print $3}')
  #duration=$(../openMP/nbody_barnes_hut $nparticles $T_FINAL | grep "Simulation took" | awk '{print $3}')

  #cuda
  #duration=$(../cuda/nbody_brute_force $nparticles $T_FINAL | grep "Simulation took" | awk '{print $3}')
  #duration=$(../cuda/nbody_barnes_hut $nparticles $T_FINAL | grep "Simulation took" | awk '{print $3}')

  #mpi
  #duration=$(mpirun -n 4 ... ../mpi/nbody_brute_force $nparticles $T_FINAL | grep "Simulation took" | awk '{print $3}')
  #duration=$(mpirun -n 4 ... ../mpi/nbody_barnes_hut $nparticles $T_FINAL | grep "Simulation took" | awk '{print $3}')

  #mpi-cuda
  #duration=$(../mpi_cuda/nbody_brute_force $nparticles $T_FINAL | grep "Simulation took" | awk '{print $3}')
  #duration=$(../mpi_cuda/nbody_barnes_hut $nparticles $T_FINAL | grep "Simulation took" | awk '{print $3}')
  
  echo "Execution time: $duration seconds"
  total_time=$(echo "$total_time + $duration" | bc)
done

# Calculate the average execution time
average_time=$(echo "scale=4; $total_time / $num_runs" | bc)
echo "Average execution time for $num_runs runs: $average_time seconds"

