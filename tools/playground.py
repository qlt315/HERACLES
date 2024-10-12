import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
xs = np.arange(-3000, 3000)/1000
plt.plot(xs, ss.erf(xs))
plt.show()

print(ss.erf(0.3))