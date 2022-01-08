import numpy as np
from scipy import stats #needed for Phi

Phi = stats.norm.cdf
def bs_call(S, E, T, r, sigma_sq, D):
    sigma = np.sqrt(sigma_sq)
    d1 = (np.log(S/E)+(r-D+(sigma_sq/2))*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    return(S*np.exp(-D*T)*Phi(d1)-E*np.exp(-r*T)*Phi(d2))