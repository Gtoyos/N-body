import subprocess

#Generate avg execution time for each algorithm

NUM_OF_RUNS = 10

for i in range(NUM_OF_RUNS):
    output = subprocess.run("../sequential/nbody_brute_force $nparticles $T_FINAL | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(float(output))
    