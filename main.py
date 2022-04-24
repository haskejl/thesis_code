import numpy as np
import sys

import bsfunc as bsf
import estimdist as est
import quadtree as qt
import comb_s_v_tree as svt

def main_quadtree_one_strike(p_in):
    nruns = 100
    N = 100
    strike = 70
    p = p_in
    # 3 months of data
    #data = open("./data/IBM_2005_18May_to_18Jul.csv")
    # 1 year of data
    #data = open("./data/IBM_2004_18Jul_to_2005_18Jul.csv")
    #lines = data.readlines()
    #data.close()

    #X_vals = np.zeros(len(lines))
    #for i in range(0,len(lines)):
    #    X_vals[i] = np.log(float(lines[i].split(",")[1].strip()))

    #data = open("./data/sp500.csv")
    #lines = data.readlines()
    #data.close()

    #X_vals = np.zeros(len(lines))
    #for i in range(0,len(lines)):
    #    X_vals[i] = np.log(float(lines[i].strip()))
    #X_vals = np.log(np.array([78.38,78.21]))
    X_vals = np.log(np.array([76.36,77.16,76.41,76.51,75.81,76.0,77.14,77.1,75.55,76.84,77.35,75.79,75.0,75.04,74.8,
        74.93,74.77,75.05,74.89,76.3,77.05,76.39,76.55,76.41,77.23,75.41,74.01,73.88,75.3,074.73,
        74.2,74.67,74.79,75.81,77.38,79.3,78.96,80.04,81.45,82.42,82.38,81.81]))
    
    s0 = 83.7
    #s0 = 1139.93
    x0 = np.log(s0) 
    res = np.zeros(nruns)
    print("Calculating cdf...")
    cdf_Ybar = est.calc_cdf(X_vals)
    sigma = qt.calc_sigma(cdf_Ybar[1][0])*cdf_Ybar[0][0]
    for i in range(1, len(cdf_Ybar[0])):
        sigma += qt.calc_sigma(cdf_Ybar[1][i])*(cdf_Ybar[0][i]-cdf_Ybar[0][i-1])
    print(sigma)
    sigma2 = np.zeros(nruns)
    for run in range(0,nruns):
        sys.stdout.write("\rPerforming run " + str(run+1) + " of " + str(nruns))
        sys.stdout.flush()

        Y_bar = est.use_cdf(cdf_Ybar[0], cdf_Ybar[1], N)
        # Use the line below to simulate constant volatility on the tree
        #Y_bar = np.repeat([np.log(0.234)], N)
        sigma2[run] = np.mean(qt.calc_sigma(Y_bar))
        res[run] = qt.calc_quad_tree_ev(x0, Y_bar, N, strike, qt.payoff_func_call, p)
    
    
    avg_sigma = np.mean(sigma2)
    bs_price_avg_sigma = bsf.bs_call(s0, strike, 42/252, 0.0343, avg_sigma, 0)
    bs_price = bsf.bs_call(s0, strike, 42/252, 0.0343, 0.234, 0)
    #bs_price_avg_sigma = bsf.bs_call(s0, strike, 29/252, 0.01, avg_sigma, 0)
    #bs_price = bsf.bs_call(s0, strike, 29/252, 0.01, 0.13, 0)

    print("\n")
    print("P=", p)
    #print("Avg sigma ", sigma)
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
    main_quadtree_one_strike(0.135)
    #main_quadtree_all_strikes()
    #svt.calc_sv_tree(10, 5, 5, np.log(10), 0.25**2)
