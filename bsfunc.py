import numpy as np
from scipy import stats #needed for Phi

Phi = stats.norm.cdf
def bs_call(S, E, T, r, sigma_sq, D):
    sigma = np.sqrt(sigma_sq)
    d1 = (np.log(S/E)+(r-D+(sigma_sq/2))*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    return(S*np.exp(-D*T)*Phi(d1)-E*np.exp(-r*T)*Phi(d2))

if __name__ == "__main__":
    s_open = 80.99
    s_high = 81.37
    days = 43
    incl_lday = 44
    p1 = bs_call(s_open, 70, days/252, 0.0343, 0.234, 0)
    p2 = bs_call(s_high, 70, days/252, 0.0343, 0.234, 0)
    p3 = bs_call(s_open, 70, incl_lday/252, 0.0343, 0.234, 0)
    p4 = bs_call(s_high, 70, incl_lday/252, 0.0343, 0.234, 0)
    print("Open w/ 43 days: " + str(p1))
    print("High w/ 43 days: " + str(p2))
    print("Open w/ Labor day: " + str(p3))
    print("Close w/ Labor day: " + str(p4))