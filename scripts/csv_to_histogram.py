import numpy as np
from numpy import genfromtxt
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

data = genfromtxt('../data/reflectivity.csv', delimiter=',')
X2 = np.sort(data)
N = len(data)
F2 = np.array(range(N))/float(N)

plt.plot(X2, F2)

plt.show()