import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


N1 = 20
p1 = 0.5
N2 = 40
p2 = 0.7
x1 = np.arange(0, N1, 1)
y1 = comb(N1, x1)*(p1**x1)*((1 - p1)**(N1 - x1))        #Bernoulli Distribution with N = 20 and p = 0.5
x2 = np.arange(0, N2)
y2 = comb(N2, x2)*(p2**x2)*((1 - p2)**(N2 - x2))        #Bernoulli Distribution with N = 40 and p = 0.7

plt.xlim(xmin=0, xmax=N2)
plt.xlabel('k', size = 10)
plt.ylabel('Probability', size = 10)
plt.title('Binomial Distribution')
plt.scatter(x1, y1,color='b', alpha=0.5, label = 'N = 20, p = 0.5')
plt.scatter(x2, y2,color='r', alpha=0.5, label = 'N = 40, p = 0.7')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
