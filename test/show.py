import matplotlib.pyplot as plt

X=[]
Y=[]

with open('openMP_execution_times_brute_force.dat', 'r') as file:
	for line in file:
		x,y=map(float, line.split())
		X.append(x)
		Y.append(y)
		
plt.plot(X,Y)

plt.xlabel('number of points')
plt.ylabel('execution time')
plt.title('openMP brute')

plt.show()
