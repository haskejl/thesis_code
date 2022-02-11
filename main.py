import xlsxwriter
import numpy as np

import bsfunc as bsf
import estimdist as est
import quadtree as qt
import comb_s_v_tree as svt

def main_quadtree_one_strike():
    #workbook = xlsxwriter.Workbook("quadtree_output.xlsx")
    #Previous year's IBM close dates (Jul 19, 2004 to July 18, 2005)
    #data = open("./data/19072004_19072005_IBM.csv")
    #data = open("./data/10days_IBM.csv")
    #lines = data.readlines()
    #data.close()

    #X_vals = np.zeros(len(lines)-1)
    #for i in range(0,len(lines)-1):
    #    X_vals[i] = np.log(float(lines[i+1].split(",")[4].strip()))
    nruns = 100
    N = 100
    #strikes = np.array([70, 75]) #smaller array for testing
    strike = 70
    X_vals = np.log(np.array([78.09,80.25])) #HL Jul 18, 2005
    #X_vals = np.log(np.array([78.38,78.21])) #OC Jul 18, 2005 
    s0 = 83.7 # Price given on p. 36
    x0 = np.log(s0) 
    res = np.zeros(nruns)
    print("Calculating cdf...")
    cdf_Ybar = est.calc_cdf(X_vals)
    #row = 1
    #col = 0
    #worksheet = workbook.add_worksheet("Results")
    #worksheet.write(0, 0, "CDF")
    #worksheet.write(0, 1, "Y Bar")
    
    #for i in range(0,len(cdf_Ybar[0])):
    #    worksheet.write(row, col, cdf_Ybar[0][i])
    #    worksheet.write(row, col+1, cdf_Ybar[1][i])
    #    row += 1

    for run in range(0,nruns):
        print("Run #", run+1, "/", nruns)

        Y_bar = est.gen_Y_bars(cdf_Ybar[0], cdf_Ybar[1], N)
        res[run] = qt.calc_quad_tree_ev(x0, Y_bar, N, E=strike)
        #print()
    
    bs_price = bsf.bs_call(s0, strike, 42/252, 0.0343, 0.234**2, 0)
    #row = 1
    #col = 4
    #worksheet.write(0, 4, "Overall EV:")
    #worksheet.write(0, 5, np.mean(res))
    #worksheet.write(1, 4, "BS Price:")
    #worksheet.write(1, 5, bs_price)
    print()
    print("For strike ", strike, ":")
    print("Overall EV for strike, = ", np.mean(res))
    print("BS Price for assumptions = ", bs_price)
    #workbook.close()
    return 0

def main_quadtree_all_strikes():
    nruns = 100
    N = 100
    #strikes = np.array([70, 75]) #smaller array for testing
    strikes = np.array([60, 70, 75, 80, 85, 90, 95])
    X_vals = np.log(np.array([78.09,80.25])) #HL Jul 18, 2005
    #X_vals = np.log(np.array([78.38,78.21])) #OC Jul 18, 2005 
    s0 = 83.7 # Price given on p. 36
    x0 = np.log(s0) 
    res = np.zeros((nruns, len(strikes)))
    print("Calculating cdf...")
    cdf_Ybar = est.calc_cdf(X_vals)

    for run in range(0,nruns):
        print("Run #", run+1, "/", nruns)
        for strike in range(0, len(strikes)):
            Y_bar = est.gen_Y_bars(cdf_Ybar[0], cdf_Ybar[1], N)
            res[run,strike] = qt.calc_quad_tree_ev(x0, Y_bar, N, E=strikes[strike])
        #print()
    
    bs_price = bsf.bs_call(s0, strikes, 42/252, 0.0343, 0.234**2, 0)

    print()
    for i in range(0,len(strikes)):
        print("For strike ", strikes[i], ":")
        print("Overall EV for strike: = ", np.mean(res[:,i]))
        print("BS Price for assumptions = ", bs_price[i])
        print()

    return 0

if __name__ == "__main__":
    main_quadtree_one_strike()
    #main_quadtree_all_strikes()
    #svt.calc_sv_tree(10, 5, 5, np.log(10), 0.25**2)