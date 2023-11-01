import subprocess

#Generate avg execution time for each algorithm

NUM_OF_RUNS = 10

sequential_results = []

for p in range(100,8000,500):
    s=0
    for i in range(NUM_OF_RUNS):
        output = subprocess.run("../sequential/nbody_brute_force $nparticles $T_FINAL | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
        s += float(output)
    sequential_results.append((p,s/NUM_OF_RUNS))
print(sequential_results)
        