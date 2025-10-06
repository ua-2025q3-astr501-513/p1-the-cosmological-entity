from henkel_pk import CorrfuncCalculator, HenkelTransform
import matplotlib.pyplot as plt


calc_xi = CorrfuncCalculator(
    datafile="mock_catalog.npz",
    sample_frac=0.01   # fraction of data used
)

r, xi, xi_err = calc_xi.compute_xi()
calc_xi.save_xi()
calc_xi.plot_xi()


calc_pk_from_xi = HenkelTransform(datafile="xi_0.01.txt", sample_frac=0.01, boxsize=1380.0, nbar=3e-3)

k_xi, pk_xi, sigma_Pk = calc_pk_from_xi.compute_pk()
calc_pk_from_xi.save_pk()
calc_pk_from_xi.plot_pk()


