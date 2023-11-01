#!/usr/bin/python

import subprocess
import numpy as np
import argparse

#Generate avg execution time for each algorithm
#Make sure to complie everything before running this script
#Run this script from the test folder

parser = argparse.ArgumentParser("benchmark.py")
parser.add_argument("algorithm", help="Algorithm to test.", type=str)
parser.add_argument("-n", "--nparticles", help="Max number of particles to test", type=int, default=8000)
args = parser.parse_args()

NUM_OF_RUNS = 10
MAX_PARTICLES = args.nparticles
STEP_SIZE = 500

if args.algorithm == "sequential":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("../sequential/nbody_brute_force "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")

if args.algorithm == "sequentialv2":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("../sequential/nbody_brute_force_v2 "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")
            
if args.algorithm == "seqbarnes":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("../sequential/nbody_barnes_hut "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")

if args.algorithm == "omp_brut_eff":
    sequential_results = []
    for p in [500,1000,3000,6000]:
        for th in [2**x for x in range(1,12) if 2**x <= p]:
            r =[]
            for i in range(NUM_OF_RUNS):
                output = subprocess.run("export OMP_NUM_THREADS="+str(th)+"; ../openMP/nbody_brute_force "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
                r.append(float(output))
                #stop is std is small enough
                if i > 2 and np.std(r) <  np.mean(r)*0.1:
                    break
                print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
            sequential_results.append((th,p,np.mean(r)))
            print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) +" "+ str(item[2])+"\n")

if args.algorithm == "omp_barnes_eff":
    sequential_results = []
    for p in [500,1000,3000,6000]:
        for th in [2**x for x in range(1,12) if 2**x <= p]:
            r =[]
            for i in range(NUM_OF_RUNS):
                output = subprocess.run("export OMP_NUM_THREADS="+str(th)+"; ../openMP/nbody_barnes_hut "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
                r.append(float(output))
                #stop is std is small enough
                if i > 2 and np.std(r) <  np.mean(r)*0.1:
                    break
                print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
            sequential_results.append((th,p,np.mean(r)))
            print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) +" "+ str(item[2])+"\n")

if args.algorithm == "openMPbrute":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("export OMP_NUM_THREADS=64; ../openMP/nbody_brute_force "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")
            
if args.algorithm == "openMPbarnes":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("sleep 0.5; export OMP_NUM_THREADS=64; ../openMP/nbody_barnes_hut "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) < np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")
            
if args.algorithm == "mpi_brute_nodes":
    sequential_results = []
    for p in [500,1000,3000,6000]:
        for nodes in [3*x for x in range(0,12)]:
            r =[]
            for i in range(NUM_OF_RUNS):
                output = subprocess.run("mpirun -np "+str(nodes)+" --hostfile hosts ../mpi/nbody_brute_force "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
                r.append(float(output))
                #stop is std is small enough
                if i > 2 and np.std(r) <  np.mean(r)*0.1:
                    break
                print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
            sequential_results.append((nodes,p,np.mean(r)))
            print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) +" "+ str(item[2])+"\n")
            
if args.algorithm == "mpi_brutev3_nodes":
    sequential_results = []
    for p in [500,1000,3000,6000]:
        for nodes in [3*x for x in range(0,12)]:
            r =[]
            for i in range(NUM_OF_RUNS):
                output = subprocess.run("mpirun -np "+str(nodes)+" --hostfile hosts ../mpi/nbody_brute_force_v3 "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
                r.append(float(output))
                #stop is std is small enough
                if i > 2 and np.std(r) <  np.mean(r)*0.1:
                    break
                print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
            sequential_results.append((nodes,p,np.mean(r)))
            print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) +" "+ str(item[2])+"\n")
            
if args.algorithm == "mpi_barnes_nodes":
    sequential_results = []
    for p in [500,1000,3000,6000]:
        for nodes in [3*x for x in range(0,12)]:
            r =[]
            for i in range(NUM_OF_RUNS):
                output = subprocess.run("mpirun -np "+str(nodes)+" --hostfile hosts ../mpi/nbody_barnes_hut "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
                r.append(float(output))
                #stop is std is small enough
                if i > 2 and np.std(r) <  np.mean(r)*0.1:
                    break
                print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
            sequential_results.append((nodes,p,np.mean(r)))
            print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) +" "+ str(item[2])+"\n")
            
if args.algorithm == "cuda_brute":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("../cuda/nbody_brute_force "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")
            
if args.algorithm == "cuda_brutev2":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("../cuda/nbody_brute_force_v2 "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")
            
if args.algorithm == "cuda_brutev3":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("../cuda/nbody_brute_force_v3 "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")

if args.algorithm == "cudabrutev4":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("../cuda/nbody_brute_force_v4 "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")
            
if args.algorithm == "mpibrute":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("mpirun -np 20 --hostfile hosts ../mpi/nbody_brute_force "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")
            
if args.algorithm == "mpibrutev3":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("mpirun -np 20 --hostfile hosts ../mpi/nbody_brute_force_v3 "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")
            
if args.algorithm == "mpibarnes":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("mpirun -np 20 --hostfile hosts ../mpi/nbody_barnes_hut "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")
            
if args.algorithm == "mpicuda_brute":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("mpirun -np 20 --hostfile hosts_gpu ../mpi_cuda/nbody_brute_force "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")
            
if args.algorithm == "mpicuda_brutev3":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("mpirun -np 20 --hostfile hosts_gpu ../mpi_cuda/nbody_brute_force_v3 "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")

if args.algorithm == "mpiomp_brute":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("export OMP_NUM_THREADS=64; mpirun -np 20 --hostfile hosts ../mpi_openMP/nbody_brute_force "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")
            
if args.algorithm == "mpiomp_brutev3":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("export OMP_NUM_THREADS=64; mpirun -np 20 --hostfile hosts ../mpi_openMP/nbody_brute_force_v3 "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")
            
if args.algorithm == "mpiomp_barnes":
    sequential_results = []
    for p in range(100,MAX_PARTICLES,STEP_SIZE):
        r =[]
        for i in range(NUM_OF_RUNS):
            output = subprocess.run("export OMP_NUM_THREADS=64; mpirun -np 20 --hostfile hosts ../mpi_openMP/nbody_barnes_hut "+str(p)+" 2 | grep 'Simulation took' | awk '{print $3}'", shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8')
            r.append(float(output))
            #stop is std is small enough
            if i > 2 and np.std(r) <  np.mean(r)*0.1:
                break
            print("nparticles="+str(p),i,"/",NUM_OF_RUNS)
        sequential_results.append((p,np.mean(r)))
        print(sequential_results)
    #save results
    with open('bench_results/'+args.algorithm+'_results.txt', 'w') as f:
        for item in sequential_results:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")