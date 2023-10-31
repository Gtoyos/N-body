#!/bin/bash

# Define the range of particle numbers and other parameters
min_particles=1000
max_particles=5000
step=500
num_runs=5

# Output files
output="execution_timese.dat"
"


> "$output"

# Loop over different numbers of particles
for ((nparticles = min_particles; nparticles <= max_particles; nparticles += step)); do
  echo "Simulating with $nparticles particles..."

  # Run the nbody_brute_force program and collect execution time
  total_time_brute_force=0
  for ((i = 1; i <= num_runs; i++)); do
  #sequential
  duration=$(../sequential/nbody_brute_force $nparticles | grep "Simulation took" | awk '{print $3}')
  #duration=$(../sequential/nbody_barnes_hut $nparticles | grep "Simulation took" | awk '{print $3}')

  #openMP
  duration=$(../openMP/nbody_brute_force $nparticles $T_FINAL | grep "Simulation took" | awk '{print $3}')
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

    total_time=$(echo "$total_time + $duration" | bc)
  done
  average_time=$(echo "scale=4; $total_time / $num_runs" | bc)
  

  # Append the results to output files
  echo "$nparticles $average_time" >> "$output"
done

# Plot the data using GNU Plot
gnuplot << EOL
set term x11 0
set title "Number of Particles vs. Execution Time"
set xlabel "Number of Particles"
set ylabel "Average Execution Time (seconds)"
set key right top
plot "$output" using 1:2 with lines title "Brute Force"

pause -1 "Press Enter to close the plot window"
EOL

