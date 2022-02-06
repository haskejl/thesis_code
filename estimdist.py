import numpy as np

#phi as defined on page 10
def calc_phi(x):
    # Written using multiplication to allow for vectorized operations
    return ((1-abs(x))*(x > -1)*(x < 1))

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
    #M selected based on page 23
    M = 300
    dt = h/M
    sqrt_h_M = np.sqrt(dt)
    #alpha, mu, and nu are defined on page 25 from the IBM specific data
    alpha = 11.85566
    mu = 0.04588
    nu = 0.9345938
    
    Y_prime = np.zeros(M)
    X_prime = np.zeros(M)
    X_prime[0] = X
    Y_prime[0] = Y
    
    #Formula 3.2
    u = np.random.normal(size = M)
    u_prime = np.random.normal(size = M)

    h_M_alpha = dt*alpha
    # psi is currently constant
    sqrt_h_M_psi_u = sqrt_h_M*calc_psi(Y_prime[0])*u
    sqrt_h_M_u_prime = sqrt_h_M*u_prime
    
    for i in range(1,M):
        Y_prime[i] = Y_prime[i-1] + h_M_alpha*(nu-Y_prime[i-1])+sqrt_h_M_psi_u[i-1]
        sig = calc_sigma(Y_prime[i-1])
        X_prime[i] = X_prime[i-1] + dt*(mu-sig**2/2)+sig*sqrt_h_M_u_prime[i-1]

    return (X_prime[M-1], Y_prime[M-1])

def use_cdf(cdf, Y_primes):
    rand_num = np.random.uniform()
    res = 0
    start = 0

    for i in range(start,len(cdf)):
        if rand_num <= cdf[i]:
            res = i
            break
    return Y_primes[res]

#The selection step takes in n (X',Y') pairs and the actual value of X for that time moment
#it then generates a cdf for the discrete distribution and uses a uniformly distributed random number from the numpy library to select a value of Y' from this distribution
#and returns C and a value of Y' to start the next mutation step
def selection_step(X_primes, Y_primes, x, n):
    phis = calc_phi(X_primes-x)

    # Remove leading values with 0 probability
    while phis[0] == 0:
        phis = np.delete(phis, 0)
        Y_primes = np.delete(Y_primes, 0)
    n = len(Y_primes)
    cdf = np.zeros(n)
    cdf[0] = phis[0]
    i = 1
    while i < n:
        # Remove any values with 0 probability
        if phis[i] == 0:
            phis = np.delete(phis, i)
            Y_primes = np.delete(Y_primes, i)
            n -= 1
        else:
            cdf[i] = cdf[i-1] + phis[i]
            i += 1
    # Holding out on the division by C until here allows the operation to be vectorized,
    #  and to avoid unnecessary 0/C
    cdf = cdf/sum(phis)
    # This gets rid of floating point error, maybe not the best way to do it,
    #  but it guarantees a probability of 1. Typically these probabilities are on the 
    #  order of 10^-5 to 10^-3 whereas the error is much smaller 10^-16
    cdf[n-1] = round(cdf[n-1], 0)
    return (use_cdf(cdf, Y_primes), cdf)

def calc_cdf(X):
    #n is selected from p.23 (1000)
    n = 1000
    #K is the length of the historical data (in days for daily data)
    K = len(X)
    
    #nu from page 25 in the IBM data
    nu = 0.9345938

    X_prime_out = np.zeros(n)
    Y_prime_out = np.zeros(n)
    cdf_out = np.zeros(n)
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
    # Can hold out on this calculation since sel_res isn't scoped to the loop & it isn't required to be set
    #  every time to calculate the next result
    cdf_out = sel_res[1]
    return(cdf_out, Y_prime_out)

def gen_Y_bars(cdf, Y_prime, N):
    Y_bar = np.zeros(N)
    for i in range(0,N):
        Y_bar[i] = use_cdf(cdf, Y_prime)
    return Y_bar

if __name__ == "__main__":
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
    calc_cdf(X_vals)