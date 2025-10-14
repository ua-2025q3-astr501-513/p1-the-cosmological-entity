import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/mock_catalog_v2.npz")
positions = data['pos']
weights = data['weight']

print(positions[0:10])

# power spectrum used to create mock catalog
data = np.loadtxt("data/power_spectrum_v2.txt", skiprows=1)
k = data[:, 0]
P_k = data[:, 1]

plt.loglog(k,P_k)
plt.show()
