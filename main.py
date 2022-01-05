import xlsxwriter
import numpy as np

import quadtree as qt

def main():
    workbook = xlsxwriter.Workbook("quadtree_output.xlsx")
    #Previous year's IBM close dates (Jul 19, 2004 to July 18, 2005)
    #data = open("./data/19072004_19072005_IBM.csv")
    #data = open("./data/10days_IBM.csv")
    #lines = data.readlines()
    #data.close()

    #X_vals = np.zeros(len(lines)-1)
    #for i in range(0,len(lines)-1):
    #    X_vals[i] = float(lines[i+1].split(",")[4].strip())
    nruns = 100
    N = 100
    X_vals = np.log(np.array([78.09,80.25])) #HL Jul 18, 2005
    #X_vals = np.array([78.38,78.21]) #OC Jul 18, 2005 
    s0 = 80.99 # open price for Jul 19, 2005 O: 80.99, H:81.37, L: 80.02, C: 80.02
    x0 = np.log(s0) 
    res = np.zeros(nruns)
    print("Calculating cdf...")
    cdf_Ybar = qt.calc_cdf(X_vals)
    row = 1
    col = 0
    worksheet = workbook.add_worksheet("Results")
    worksheet.write(0, 0, "CDF")
    worksheet.write(0, 1, "Y Bar")
    
    for i in range(0,len(cdf_Ybar[0])):
        worksheet.write(row, col, cdf_Ybar[0][i])
        worksheet.write(row, col+1, cdf_Ybar[1][i])
        row += 1

    for run in range(0,nruns):
        print("Run #", run+1, "/", nruns)
        Y_bar = qt.gen_Y_bars(cdf_Ybar[0], cdf_Ybar[1], N)
        res[run] = qt.calc_quad_tree_ev(x0, Y_bar, N, E=70)
        #print()
    
    bs_price = qt.bs_call(s0, 70, 43/252, 0.0343, 0.234, 0)
    row = 1
    col = 4
    worksheet.write(0, 4, "Overall EV:")
    worksheet.write(0, 5, np.mean(res))
    worksheet.write(1, 4, "BS Price:")
    worksheet.write(1, 5, bs_price)
    print()
    print("Overall EV = ", np.mean(res))
    print("BS Price for assumptions = ", bs_price)
    workbook.close()
    return 0

if __name__ == "__main__":
    main()