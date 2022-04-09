import numpy as np

#alpha, beta, mu, and nu are defined on page 25 from the IBM specific data
alpha = 11.85566
beta = 4.13415
mu = 0.04588
nu = 0.9345938
#S&P
#alpha = 50.0
#beta = 1.0
#nu = -4.38
#mu = 0.04


#phi as defined on page 10
def calc_phi(x):
    # Written using multiplication to allow for vectorized operations
    return ((1-abs(x))*(x > -1)*(x < 1))

#psi as defined on page 23
def calc_psi(y):
    return beta

#sigma as defined on page 23
def calc_sigma(y):
    return np.exp(-abs(y))

#The mutation step takes in an (X,Y) pair correspoinding to t_i
#and returns an (X',Y') pair corresponding to t_{i+1}
def mutation_step(X, Y, n):
    assert n == len(Y) or len(Y)==1, "len(Y) should equal 1 or n" + "but len(Y)=" + str(len(Y)) + " and n=" + str(n)
    h = 1.0/252.0 # day
    M = 300 # Value from p. 23
    
    X_prime = np.zeros(n)
    Y_prime = np.zeros(n)
    X_prime = X
    Y_prime = Y
    
    #Formula 3.2
    u = np.random.normal(size = (M,n))
    u_prime = np.random.normal(size = (M,n))
    h_M = h/M
    sqrt_h_M = np.sqrt(h_M)
    h_M_alpha = h_M*alpha
    # psi is currently constant as beta
    sqrt_h_M_psi_u = sqrt_h_M*beta*u
    sqrt_h_M_u_prime = sqrt_h_M*u_prime
    
    for i in range(1,M+1):
        sig = calc_sigma(Y_prime)
        Y_prime = Y_prime + h_M_alpha*(nu-Y_prime)+sqrt_h_M_psi_u[i-1]
        X_prime = X_prime + h_M*(mu-sig**2/2)+sig*sqrt_h_M_u_prime[i-1]

    return (X_prime, Y_prime)

def use_cdf(cdf, Y_primes, n):
    rand_num = np.random.uniform(size=n)
    res = 0
    start = 0
    Y_primes_out = np.zeros(n)

    for i in range(0, n):
        for j in range(start,len(cdf)):
            if rand_num[i] <= cdf[j]:
                Y_primes_out[i] = Y_primes[j]
                break
    return Y_primes_out

#The selection step takes in n (X',Y') pairs and the actual value of X for that time moment
#it then generates a cdf for the discrete distribution and uses a uniformly distributed random number from the numpy library to select a value of Y' from this distribution
#and returns C and a value of Y' to start the next mutation step
def selection_step(X_primes, Y_primes, x, n):
    phis = calc_phi(X_primes-x)
    #print("For x: ", x)
    #print(phis)
    # Remove leading values with 0 probability
    while phis[0] == 0:
        phis = np.delete(phis, 0)
        Y_primes = np.delete(Y_primes, 0)
    n_temp = len(Y_primes)
    cdf = np.zeros(n_temp)
    cdf[0] = phis[0]
    i = 1
    while i < n_temp:
        # Remove any values with 0 probability
        if phis[i] == 0:
            phis = np.delete(phis, i)
            Y_primes = np.delete(Y_primes, i)
            n_temp -= 1
        else:
            cdf[i] = cdf[i-1] + phis[i]
            i += 1
    # Holding out on the division by C until here allows the operation to be vectorized,
    #  and to avoid unnecessary 0/C
    cdf = cdf/sum(phis)
    # This gets rid of floating point error, maybe not the best way to do it,
    #  but it guarantees a probability of 1. Typically these probabilities are on the 
    #  order of 10^-5 to 10^-3 whereas the error is much smaller 10^-16
    cdf[n_temp-1] = round(cdf[n_temp-1], 0)
    return (use_cdf(cdf, Y_primes, n), cdf)

def calc_cdf(X):
    #n is selected from p.23 (1000)
    n = 1000
    #K is the length of the historical data (in days for daily data)
    K = len(X)

    X_prime_out = np.zeros(n)
    Y_prime_out = np.zeros(n)
    cdf_out = np.zeros(n)
    #Generate n (X',Y') pairs
    X_prime_out, Y_prime_out = mutation_step(X[0], np.array([nu]), n)
    
    #Use the n (X',Y') pairs to pick a realized value of Y' for the next time step
    Y_t_i, cdf_out = selection_step(X_prime_out, Y_prime_out, X[1], n)
    
    # Range is correct, step 1 goes from 0 to 1, step K goes from K-2 to K-1
    #  transformed the 
    for i in range(1,K-1):
        #Generate n (X',Y') pairs
        X_prime_out, Y_prime_out = mutation_step(X[i], Y_t_i, n)
        
        #Use the n (X',Y') pairs to pick a realized value of Y' for the next time step
        # This is what makes the upper limit K-2 correct, if range(1, K) is used
        # X[i+1] is an index error
        Y_t_i, cdf_out = selection_step(X_prime_out, Y_prime_out, X[i+1], n)

    return(cdf_out, Y_prime_out)
