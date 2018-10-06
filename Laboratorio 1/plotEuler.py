import matplotlib.pyplot as plt
import numpy as np

file = open("euler.txt", 'r')
deltaT = 1

for line in file:
	line = line.split(" ")
	for i in range(len(line)):
		line[i] = float(line[i])
	deltaT *= 0.1
	y = np.linspace(0, 10, round(10 / deltaT,1))
	plt.plot(y, line, 'ro')
	plt.show()
