import xlsxwriter
import numpy as np

import bsfunc as bsf
import estimdist as est
import quadtree as qt
import comb_s_v_tree as svt

def main_quadtree_one_strike():
    nruns = 100
    N = 100
    strike = 70
    p = 0.145
    X_vals = np.log(np.array([83.94,81.68])) #HL Jul 18, 2005
    #X_vals = np.log(np.array([81.99,83.70 or 81.81])) #OC Jul 18, 2005 
    s0 = 83.7 # Price given on p. 36
    x0 = np.log(s0) 
    res = np.zeros(nruns)
    print("Calculating cdf...")
    cdf_Ybar = est.calc_cdf(X_vals)

    for run in range(0,nruns):
        print("Run #", run+1, "/", nruns)

        Y_bar = est.use_cdf(cdf_Ybar[0], cdf_Ybar[1], N)
        res[run] = qt.calc_quad_tree_ev(x0, Y_bar, N, strike, qt.payoff_func_call, p)
    
    bs_price = bsf.bs_call(s0, strike, 42/252, 0.0343, 0.234, 0)

    print()
    print("For strike ", strike, ":")
    print("Overall EV for strike, = ", np.mean(res))
    print("BS Price for assumptions = ", bs_price)
    return 0

def main_quadtree_all_strikes():
    nruns = 100
    N = 100
    #strikes = np.array([70, 75]) #smaller array for testing
    strikes = np.array([60, 70, 75, 80, 85, 90, 95])
    p = 0.14
    X_vals = np.log(np.array([83.94,81.68])) #HL Jul 18, 2005
    #X_vals = np.log(np.array([78.38,78.21])) #OC Jul 18, 2005 
    s0 = 83.7 # Price given on p. 36
    x0 = np.log(s0) 
    res = np.zeros((nruns, len(strikes)))
    print("Calculating cdf...")
    cdf_Ybar = est.calc_cdf(X_vals)

    for run in range(0,nruns):
        print("Run #", run+1, "/", nruns)
        for strike in range(0, len(strikes)):
            Y_bar = est.use_cdf(cdf_Ybar[0], cdf_Ybar[1], N)
            res[run,strike] = qt.calc_quad_tree_ev(x0, Y_bar, N, strikes[strike], qt.payoff_func_call, p)
        #print()
    
    bs_price = bsf.bs_call(s0, strikes, 42/252, 0.0343, 0.234, 0)

    print()
    for i in range(0,len(strikes)):
        print("For strike ", strikes[i], ":")
        print("Overall EV for strike: = ", np.mean(res[:,i]))
        print("BS Price for assumptions = ", bs_price[i])
        print()

    return 0

if __name__ == "__main__":
    #main_quadtree_one_strike()
    main_quadtree_all_strikes()
    #svt.calc_sv_tree(10, 5, 5, np.log(10), 0.25**2)