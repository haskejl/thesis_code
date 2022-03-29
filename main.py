import numpy as np
import sys

import bsfunc as bsf
import estimdist as est
import quadtree as qt
import comb_s_v_tree as svt

def main_quadtree_one_strike():
    nruns = 100
    N = 100
    strike = 70
    p = 0.14
    # 3 months of data
    data = open("./data/IBM_2005_18May_to_18Jul.csv")
    # 1 year of data, currently not working
    #data = open("./data/IBM_2004_18Jul_to_2005_18Jul.csv")
    lines = data.readlines()
    data.close()

    X_vals = np.zeros(len(lines))
    for i in range(0,len(lines)):
        X_vals[i] = np.log(float(lines[i].split(",")[1].strip()))
    #X_vals = np.repeat(83.7, 10)
    
    s0 = 83.7
    x0 = np.log(s0) 
    res = np.zeros(nruns)
    print("Calculating cdf...")
    cdf_Ybar = est.calc_cdf(X_vals)
    sigma = np.zeros(nruns)
    for run in range(0,nruns):
        sys.stdout.write("\rPerforming run " + str(run+1) + " of " + str(nruns))
        sys.stdout.flush()

        Y_bar = est.use_cdf(cdf_Ybar[0], cdf_Ybar[1], N)
        # Use the line below to simulate constant volatility on the tree
        #Y_bar = np.array([np.log(0.234)], N)
        sigma[run] = np.mean(qt.calc_sigma(Y_bar))
        res[run] = qt.calc_quad_tree_ev(x0, Y_bar, N, strike, qt.payoff_func_call, p)
    
    avg_sigma = np.mean(sigma)
    bs_price_avg_sigma = bsf.bs_call(s0, strike, 42/252, 0.0343, avg_sigma, 0)
    bs_price = bsf.bs_call(s0, strike, 42/252, 0.0343, 0.234, 0)

    print("\n")
    print("For strike ", strike, " with avg sigma ", avg_sigma, ":")
    print("Overall EV for strike = ", np.mean(res))
    print("BS Price with Avg Sigma = ", bs_price_avg_sigma)
    print("BS Price for assumptions = ", bs_price)
    print()
    return 0

def main_quadtree_all_strikes():
    nruns = 100
    N = 100
    #strikes = np.array([70, 75]) #smaller array for testing
    strikes = np.array([60, 70, 75, 80, 85, 90, 95])
    p = np.array([0.14])#np.linspace(1/12, 1/6, 20)
    #X_vals = np.array([83.94,81.68]) #HL Jul 18, 2005
    X_vals = np.log(np.array([78.38,78.21])) #OC Jul 18, 2005 
    s0 = 83.7 # Price given on p. 36
    x0 = np.log(s0) 
    res = np.zeros((nruns, len(strikes), len(p)))
    print("Calculating cdf...")
    cdf_Ybar = est.calc_cdf(X_vals)
    t_len = len(strikes)*len(p)
    for i in range(0, len(cdf_Ybar[0])):
        print(cdf_Ybar[0][i], ",", cdf_Ybar[1][i])

    for run in range(0,nruns):
        for strike in range(0, len(strikes)):
            for p_val in range(0, len(p)):
                progress = int(np.floor((strike*len(p) + p_val)/t_len*50))
                blanks = 50-progress
                sys.stdout.write("\rPerforming run " + str(run+1) + " of " + str(nruns) + " [" + u"\u25A0"*progress + " "*blanks + "]")
                sys.stdout.flush()
                Y_bar = est.use_cdf(cdf_Ybar[0], cdf_Ybar[1], N)
                res[run,strike,p_val] = qt.calc_quad_tree_ev(x0, Y_bar, N, strikes[strike], qt.payoff_func_call, p[p_val])
        #print()
    
    bs_price = bsf.bs_call(s0, strikes, 42/252, 0.0343, 0.234, 0)

    print("\n")
    for i in range(0,len(strikes)):
        print("For strike ", strikes[i], ":")
        print("                    p: = ", p)
        print("Overall EV for strike: = ", np.mean(res[:,i]))
        print("BS Price for assumptions = ", bs_price[i])
        print()

    return 0

if __name__ == "__main__":
    main_quadtree_one_strike()
    #main_quadtree_all_strikes()
    #svt.calc_sv_tree(10, 5, 5, np.log(10), 0.25**2)
