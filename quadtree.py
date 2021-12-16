import numpy as np

#phi as defined on page 10
def calc_phi(x):
    if x > -1 and x < 1:
        return 1-abs(x)
    return 0

#psi as defined on page 23
def calc_psi(y):
    #beta as defined on page 25 from the IBM specific data
    beta = 4.13415
    return beta

#sigma as defined on page 23
def calc_sigma(y):
    return np.exp(-abs(y))

#The mutation step takes in an (X,Y) pair correspoinding to t_i
#and returns an (X',Y') pair corresponding to t_{i+1}
def mutation_step(X, Y):
    h = 1 #day
    #M selected based on page 10
    M = 1000
    #alpha, mu, and nu are defined on page 25 from the IBM specific data
    alpha = 11.85566
    mu = 0.04588
    nu = 0.9345938
    
    Y_prime = np.zeros(M)
    X_prime = np.zeros(M)
    X_prime[0] = X
    Y_prime[0] = Y
    
    #Formula 3.2
    for i in range(1,M):
        Y_prime[i] = Y_prime[i-1] + h/M*alpha*(nu-Y_prime[i-1])+np.sqrt(h/M)*calc_psi(Y_prime[i-1])*np.random.normal()
        X_prime[i] = X_prime[i-1] + h/M*alpha*(mu-calc_sigma(Y_prime[i-1])**2/2)+np.sqrt(h/M)*calc_sigma(Y_prime[i-1])*np.random.normal()

    return (X_prime[M-1], Y_prime[M-1])

def use_cdf(cdf, Y_primes, n):
    # NOTE: potential for bug here if rand_num=0 and p(Y'_0) = 0, Y'_0 will still get choosen, there's also no guarantee currently that cdf[n] = 1.0
    rand_num = np.random.uniform()
    res = 0
    for i in range(0,n):
        if rand_num <= cdf[i]:
            res = i
            break
    return Y_primes[res]

#The selection step takes in n (X',Y') pairs and the actual value of X for that time moment
#it then generates a cdf for the discrete distribution and uses a uniformly distributed random number from the numpy library to select a value of Y' from this distribution
#and returns C and a value of Y' to start the next mutation step
def selection_step(X_primes, Y_primes, x, n):
    C = 0
    for i in range(0,n):
        C += calc_phi(X_primes[i]-x)
    cdf = np.zeros(n)
    cdf[0] = calc_phi(X_primes[0]-x)/C
    for i in range(1,n):
        cdf[i] = cdf[i-1] + calc_phi(X_primes[i]-x)/C
    return (use_cdf(cdf, Y_primes, n), cdf)

def calc_Y_bar(X, N):
    #n is selected from the recommendation on page 10
    n = 10
    #K is the length of the historical data (in days for daily data)
    K = len(X)
    cdf_out = np.zeros(N)
    #nu from page 25 in the IBM data
    nu = 0.9345938
    # TODO: define data structure for the X,Y pairs?
    X_prime_out = np.zeros(n)
    Y_prime_out = np.zeros(n)
    #Generate n (X',Y') pairs
    for i in range(0,n):
        mut_res = mutation_step(X[0], nu)
        X_prime_out[i] = mut_res[0]
        Y_prime_out[i] = mut_res[1]
    
    #Use the n (X',Y') pairs to pick a realized value of Y' for the next time step
    sel_res = selection_step(X_prime_out, Y_prime_out, X[0], n)
    Y_t_i = sel_res[0]

    for i in range(1,K):
        #Generate n (X',Y') pairs
        for j in range(0,n):
            mut_res = mutation_step(X[i], Y_t_i)
            X_prime_out[j] = mut_res[0]
            Y_prime_out[j] = mut_res[1]
        
        #Use the n (X',Y') pairs to pick a realized value of Y' for the next time step
        sel_res = selection_step(X_prime_out, Y_prime_out, X[i], n)
        Y_t_i = sel_res[0]
        cdf_out = sel_res[1]
    
    Y_bar = np.zeros(N)
    for i in range(0,N):
        Y_bar[i] = use_cdf(cdf_out, Y_prime_out, n)
    return Y_bar

    

def main():
    #TODO: Read from file
    #Previous year's IBM close dates (Jul 19, 2004 to July 18, 2005)
    X_vals = np.array([85.3,86.36,85.3,86.06,84.85,85.09,85.88,85.85,86.77,87.07,86.69,85.71,85.97,85.19,83.48,83.55,84.99,83.69,82.21,83.91,84.02,84.04,85.13,84.89,85.25,84.65,84.71,85.07,84.69,84.94,84.4,84.69,84.22,84.57,84.39,84.97,85.86,86.44,86.76,86.49,86.72,86.37,86.12,85.74,85.7,85.72,84.31,83.88,84.43,84.16,84.48,84.98,85.74,86.72,87.16,87.32,88.04,87.42,86.71,86.63,86,84.98,84.78,84.85,85.92,89.37,88.82,88.1,87.39,88.43,89,90,89.5,89.75,90.11,90.47,91.2,92.38,93.28,93.37,93.37,93.61,94.79,95.32,95.92,94.89,95.46,95.1,94.45,95.11,95.28,95.46,94.72,95.5,94.24,95.88,95.76,97.08,97.67,96.1,96.65,97.51,96.67,96.45,97.31,97.33,97.45,96.2,96.55,97.02,97.61,97.72,97.5,98.3,98.18,98.3,98.58,97.75,96.7,96.5,96.2,95.78,95.68,95,95.21,94.45,94.1,94.9,93.1,93,92.38,91.79,92.19,91.95,91.98,92.89,93.42,93.86,94.3,93.54,94.51,94.53,94.13,92.7,92.76,93.3,93.57,94.33,94.62,93.75,93.27,92.32,92.1,92.64,92.8,92.58,93.3,92.92,92.41,92.37,91.6,92.13,92.35,92.41,91.51,91.9,91.38,90.65,89.86,89.28,89.51,89.5,90.52,90.7,91.04,90.6,90.68,91.38,90.44,90.32,89.57,89,88.44,87.6,86.2,85.75,84.57,83.64,76.7,76.65,75.48,72.01,74.03,74.21,74.61,75.43,77.05,75.91,76.38,76.51,76.47,77.08,75.5,75.26,74.98,73.3,73.28,72.62,73.16,74.34,74.29,76.36,77.16,76.41,76.51,75.81,76,77.14,77.1,75.55,76.84,77.35,75.79,75,75.04,74.8,74.93,74.77,75.05,74.89,76.3,77.05,76.39,76.55,76.41,77.23,75.41,74.01,73.88,75.3,74.73,74.2,74.67,74.79,75.81,77.38,79.3,78.96,80.04,81.45,82.42,82.38,81.81])
    #X_vals = np.array([1.3,2.3,3.3,4.8,1.2,3.4,1.1]) #used for testing
    N = 10
    Y_bar = calc_Y_bar(X_vals, N)
    print(Y_bar)
    return 0

if __name__ == "__main__":
    main()
