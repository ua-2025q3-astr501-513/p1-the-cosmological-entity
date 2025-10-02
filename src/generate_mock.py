import numpy as np

from nbodykit.lab import *
from nbodykit import style, setup_logging

import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
plt.style.use(style.notebook)

setup_logging()


redshift = 0.55
cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
b1 = 2.0

# Evaluate over a k grid
k = np.logspace(-3, 1, 500)  # h/Mpc
Pk_lin = Plin(k)             # units (Mpc/h)^3
Pk_true = b1**2 * Pk_lin     # biased spectrum

# Save
np.savetxt("Pk_true.txt", np.c_[k, Pk_true])

cat = LogNormalCatalog(Plin=Plin, nbar=3e-3, BoxSize=1380., Nmesh=256, bias=b1, seed=42)


positions = cat['Position'].compute()
weights = cat['Weight'].compute()

# save to a text file
np.savez_compressed("mock_catalog.npz", pos=positions, weight=weights)

# convert the catalog to the mesh, with CIC interpolation
real_mesh = cat.to_mesh(compensated=True, window='cic', position='Position')

# compute the 2d P(k,mu) power, with 5 mu bins
r = FFTPower(real_mesh, mode='2d', Nmu=5)
pkmu = r.power

# plot the biased linear power spectrum
k = np.logspace(-2, 0, 512)
P_k = b1**2 * Plin(k)

# stack k and P(k) into one array
data = np.column_stack([k, P_k])
# save to text
np.savetxt("power_spectrum1.txt", data, header="k Pk")

# plot each mu bin
for i in range(pkmu.shape[1]):
    Pk = pkmu[:,i]
    label = r'$\mu$=%.1f' %pkmu.coords['mu'][i]
    plt.loglog(Pk['k'], Pk['power'].real - Pk.attrs['shotnoise'], label=label)

plt.loglog(k,P_k,  c='k', label=r'$b_1^2 P_\mathrm{lin}$')
# add a legend and axes labels
plt.legend(loc=0, ncol=2, fontsize=16)
plt.xlabel(r"$k$ [$h \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k, \mu)$ [$h^{-3} \mathrm{Mpc}^3$]")
plt.xlim(0.01, 0.6)
plt.ylim(500, 2e5)
plt.savefig("pspec_test.png")
plt.show()