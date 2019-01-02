import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
x = np.random.randn(50)
y = x * np.random.randn(50)

colors = np.random.rand(50)

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()
