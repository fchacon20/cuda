import matplotlib.pyplot as plt

file = open("output.txt", "r")
file2 = open("initialPoints.txt", "r")

x = file.readline().strip().split(',')
y = file.readline().strip().split(',')
x0 = file2.readline().strip().split(',')
y0 = file2.readline().strip().split(',')

x = list(map(float, x))
y = list(map(float, y))
x0 = list(map(float, x0))
y0 = list(map(float, y0))

plt.plot(x0, y0, "ro", markersize=8)
plt.plot(x, y, "ko", markersize=4)
plt.show()
