import numpy as np
import matplotlib.pyplot as plt

data = np.load("mock_catalog.npz")
positions = data['pos']
weights = data['weight']

print(positions[0:10])

# power spectrum used to create mock catalog
data = np.loadtxt("power_spectrum.txt", skiprows=1)
k = data[:, 0]
P_k = data[:, 1]

plt.loglog(k,P_k)
plt.show()
