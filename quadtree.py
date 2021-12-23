import numpy as np

#phi as defined on page 10
def calc_phi(x):
    # Rewritten to allow for vectorized operation
    return ((1-abs(x))*(x > -1)*(x < 1))
    '''
    if x > -1 and x < 1:
        return 1-abs(x)
    return 0'''

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
    h_M_alpha = h/M*alpha
    sqrt_h_M = np.sqrt(h/M)
    for i in range(1,M):
        Y_prime[i] = Y_prime[i-1] + h_M_alpha*(nu-Y_prime[i-1])+sqrt_h_M*calc_psi(Y_prime[i-1])*np.random.normal()
        sig = calc_sigma(Y_prime[i-1])
        X_prime[i] = X_prime[i-1] + h_M_alpha*(mu-sig**2/2)+sqrt_h_M*sig*np.random.normal()

    return (X_prime[M-1], Y_prime[M-1])

def use_cdf(cdf, Y_primes, n):
    rand_num = np.random.uniform()
    res = 0
    start = 0
    if(cdf[0] == 0):
        start = 1
    for i in range(1,n):
        if rand_num <= cdf[i]:
            res = i
            break
    return Y_primes[res]

#The selection step takes in n (X',Y') pairs and the actual value of X for that time moment
#it then generates a cdf for the discrete distribution and uses a uniformly distributed random number from the numpy library to select a value of Y' from this distribution
#and returns C and a value of Y' to start the next mutation step
def selection_step(X_primes, Y_primes, x, n):
    C = sum(calc_phi(X_primes-x))
    #for i in range(0,n):
    #    C += calc_phi(X_primes[i]-x)
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
    print("The CDF is...")
    print(cdf_out)
    Y_bar = np.zeros(N)
    for i in range(0,N):
        Y_bar[i] = use_cdf(cdf_out, Y_prime_out, n)
    return Y_bar

    

def main():
    #Previous year's IBM close dates (Jul 19, 2004 to July 18, 2005)
    data = open("./data/19072004_19072005_IBM.csv")
    lines = data.readlines()
    data.close()

    X_vals = np.zeros(len(lines)-1)
    for i in range(0,len(lines)-1):
        X_vals[i] = float(lines[i+1].split(",")[4].strip())

    #X_vals = np.array([1.3,2.3,3.3,4.8,1.2,3.4,1.1]) #used for testing
    N = 10
    print("Calculating Y bar values...")
    Y_bar = calc_Y_bar(X_vals, N)
    print(Y_bar)
    T = 10
    x0 = X_vals[-1]
    print("x0 is...")
    print(x0)
    print("Calculating the tree...")
    return 0

if __name__ == "__main__":
    main()
