#!/bin/bash

# Define the range of particle numbers and other parameters
min_particles=1000
max_particles=5000
step=500
num_runs=5

# Output files
output_brute_force="openMP_execution_times_brute_force.dat"
output_barnes_hut="openMP_execution_times_barnes_hut.dat"

# Clear the existing output files
> "$output_brute_force"
> "$output_barnes_hut"

# Loop over different numbers of particles
for ((nparticles = min_particles; nparticles <= max_particles; nparticles += step)); do
  echo "Simulating with $nparticles particles..."

  # Run the nbody_brute_force program and collect execution time
  total_time_brute_force=0
  for ((i = 1; i <= num_runs; i++)); do
    duration=$(../openMP/nbody_brute_force $nparticles | grep "Simulation took" | awk '{print $3}')
    total_time_brute_force=$(echo "$total_time_brute_force + $duration" | bc)
  done
  average_time_brute_force=$(echo "scale=4; $total_time_brute_force / $num_runs" | bc)

  # Run the nbody_barnes_hut program and collect execution time
  total_time_barnes_hut=0
  for ((i = 1; i <= num_runs; i++)); do
    duration=$(../openMP/nbody_barnes_hut $nparticles | grep "Simulation took" | awk '{print $3}')
    total_time_barnes_hut=$(echo "$total_time_barnes_hut + $duration" | bc)
  done
  average_time_barnes_hut=$(echo "scale=4; $total_time_barnes_hut / $num_runs" | bc)

  # Append the results to output files
  echo "$nparticles $average_time_brute_force" >> "$output_brute_force"
  echo "$nparticles $average_time_barnes_hut" >> "$output_barnes_hut"
done
