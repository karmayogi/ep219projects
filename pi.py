import numpy as np
import matplotlib.pyplot as plt

a = np.empty(0)
e = np.empty(0)

for j in range(0, 500):
    count = 0
    x = np.random.random_sample(2000)
    y = np.random.random_sample(2000)
    for i in range(0, 2000):
        if np.sqrt((x[i] - 0.5)**2 + (y[i] - 0.5)**2) <= 0.5:   #Check for location of point
            count += 1                                          #Incremented for each point inside the circle
    a = np.append(a, count*4.0 / 2000.0)                        #Mean value of PI for experiment
    e = np.append(e, abs((count*4/2000.0) - np.pi))             #Error for each experiment

plt.hist(a)
print("Sample Width:", np.sum(np.sqrt(e**2 / 2000)))            #Calulating the sample width of Distribution of mean value PI
print("Error in individual Experiments: \n")
print(e)
plt.xlabel('Mean value of PI')
plt.ylabel('Count')
plt.title('Distribution of mean values of PI')
plt.savefig("pi.png")
plt.show()
