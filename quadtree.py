import numpy as np
from scipy import stats #needed for Phi

class QuadTreeNode:
    def __init__(self, x, probability=0, 
    upp=None, up=None, down=None, downn=None, under_me=None):
        self.x = x
        self.upp = upp
        self.up = up
        self.down = down
        self.downn = downn
        self.under_me = under_me #allows a singularly linked list for navigating through a time moment
        self.probability = probability

    def print_mini_tree(self):
        print("Me: ", self.x)
        print(self.upp.x, " ", self.up.x, " ", self.down.x, " ", self.downn.x)

Phi = stats.norm.cdf
def bs_call(S, E, T, r, volat, D):
    sigma = np.sqrt(volat)
    d1 = (np.log(S/E)+(r-D+(volat/2))*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    return(S*np.exp(-D*T)*Phi(d1)-E*np.exp(-r*T)*Phi(d2))

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

def use_cdf(cdf, Y_primes):
    rand_num = np.random.uniform()
    res = 0
    start = 0
    # Due to the way the for loop works, a value with 0 probablility could be choosen if
    # the rng gives 0 for the random number, so the values at the beginning of the cdf array
    # with p=0 need to be "thrown out"
    while(cdf[start] == 0):
        start += 1
    for i in range(start,len(cdf)):
        if rand_num <= cdf[i]:
            res = i
            break
    return Y_primes[res]

#The selection step takes in n (X',Y') pairs and the actual value of X for that time moment
#it then generates a cdf for the discrete distribution and uses a uniformly distributed random number from the numpy library to select a value of Y' from this distribution
#and returns C and a value of Y' to start the next mutation step
def selection_step(X_primes, Y_primes, x, n):
    C = sum(calc_phi(X_primes-x))
    cdf = np.zeros(n)
    cdf[0] = calc_phi(X_primes[0]-x)/C
    for i in range(1,n):
        cdf[i] = cdf[i-1] + calc_phi(X_primes[i]-x)/C
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
        cdf_out = sel_res[1]
    return(cdf_out, Y_prime_out)

def gen_Y_bars(cdf, Y_prime, N):
    Y_bar = np.zeros(N)
    for i in range(0,N):
        Y_bar[i] = use_cdf(cdf, Y_prime)
    return Y_bar

def payoff_func(x, E, r, T):
    # TODO: does the payoff need to be discounted?
    return max(x-E,0)#*np.exp(-r*T)

def calc_quad_tree_ev(x0, Y_bar, N, E):
    # Time is 43 trading days from Jul. 19 to Sep. 16 since volatility was calculated based on days
    # If we count Jul. 19th (assume valuation on open), The 1st Monday of Sept. isn't a trading day
    T = 43/252 
    p = 0.14
    # r value is from p. 25
    r = 0.0343
    dt = T/N
    #print("Calculating the tree...")
    # Set the base node for the tree
    top_node = bottom_node = base_node = QuadTreeNode(x0, 1)
    sig = calc_sigma(Y_bar)
    add_pt = (r-sig**2/2)*dt
    mul_pt = sig*np.sqrt(dt)

    for i in range(0,N):
        #print("Running step: ", i)
        # Must be a ceiling function or j*sigma(Y)*sqrt(dt) may end up below the point
        j_upp = int(np.ceil(top_node.x/mul_pt[i]))
        j_downn = int(np.ceil(bottom_node.x/mul_pt[i]))
        j = range(j_upp+1, j_downn-3,-1)

        # Calculate all of the successors with the drift term
        nodes = {j[0]: QuadTreeNode(j*mul_pt[i]+add_pt[i])}
        for k in j:
            nodes = nodes | {k: QuadTreeNode(k*mul_pt[i]+add_pt[i])}
        # Since the nodes are shared, set up the linked list here
        for k in j[0:(len(j)-1)]:
            nodes[k].under_me = nodes[k-1]
        
        curr_node = top_node
        last_downn = None
        while(curr_node != None):
            node_j = int(np.ceil(curr_node.x/mul_pt[i]))

            d1 = curr_node.x - node_j*mul_pt[i]
            d2 = curr_node.x - (node_j-1)*mul_pt[i]
            assert d1 <= 0, "d1 = " + str(d1) + ", but should be <= 0"
            assert d2 >= 0, "d2 = " + str(d2) + ", but should be >= 0"

            p1 = 0
            p2 = 0
            p3 = 0
            p4 = 0
            if(-d1 < d2):
                q = d1/mul_pt[i]
                p4 = p
                p1 = 0.5*(1+q+q**2)-p
                p2 = 3*p-q**2
                p3 = 0.5*(1-q+q**2)-3*p
                assert p1 > 0, "p1 = " + str(p1) + ", but should be > 0 for -d1 < d2"
                assert p2 > 0, "p2 = " + str(p2) + ", but should be > 0 for -d1 < d2"
                assert p3 > 0, "p3 = " + str(p3) + ", but should be > 0 for -d1 < d2"
                assert p1 < 1, "p1 = " + str(p1) + ", but should be < 1 for -d1 < d2"
                assert p2 < 1, "p2 = " + str(p2) + ", but should be < 1 for -d1 < d2"
                assert p3 < 1, "p3 = " + str(p3) + ", but should be < 1 for -d1 < d2"
            else:
                q = d2/mul_pt[i]
                p1 = p
                p2 = 0.5*(1+q+q**2)-3*p
                p3 = 3*p-q**2
                p4 = 0.5*(1-q+q**2)-p
                assert p2 > 0, "p2 = " + str(p2) + ", but should be > 0 for -d1 > d2"
                assert p3 > 0, "p3 = " + str(p3) + ", but should be > 0 for -d1 > d2"
                assert p4 > 0, "p4 = " + str(p4) + ", but should be > 0 for -d1 > d2"
                assert p2 < 1, "p2 = " + str(p2) + ", but should be < 1 for -d1 > d2"
                assert p3 < 1, "p3 = " + str(p3) + ", but should be < 1 for -d1 > d2"
                assert p4 < 1, "p4 = " + str(p4) + ", but should be < 1 for -d1 > d2"

            #Set the successor values
            curr_node.upp = nodes[node_j+1]
            curr_node.up = nodes[node_j]
            curr_node.down = nodes[node_j-1]
            curr_node.downn = nodes[node_j-2]
            # Calculate the conditional probabilities of the successors 
            #  given each of it's predecessors has happened
            curr_node.upp.probability = curr_node.upp.probability + p1*curr_node.probability
            curr_node.up.probability = curr_node.up.probability + p2*curr_node.probability
            curr_node.down.probability = curr_node.down.probability + p3*curr_node.probability
            curr_node.downn.probability = curr_node.downn.probability + p4*curr_node.probability

            # Go to the next node in this period
            curr_node = curr_node.under_me
        # Set the top and bottom nodes for the next period
        top_node = top_node.upp
        bottom_node = bottom_node.downn
    
    # Calculate the ev of the payoff of the final row of options
    #print("Calculating EV...")
    # Move to the highest node of the payoff time moment
    curr_node = top_node
    ret_val = 0
    tot_prob = 0
    while(curr_node != None):
        tot_prob = tot_prob + curr_node.probability
        ret_val = ret_val + payoff_func(np.exp(curr_node.x), E, r, T)*curr_node.probability
        curr_node = curr_node.under_me
    #print("EV for run = ", ret_val)
    # This deviates slightly from 1 due to floating point error
    #print("Total Prob for run = ", tot_prob)
    return(ret_val)

def main():
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
    X_vals = np.array([78.09,80.25]) #HL Jul 18, 2005
    #X_vals = np.array([78.38,78.21]) #OC Jul 18, 2005
    x0 = np.log(80.99) # open price for Jul 19, 2005
    res = np.zeros(nruns)
    print("Calculating cdf...")
    cdf_Ybar = calc_cdf(X_vals)

    for run in range(0,nruns):
        print("Run #", run+1, "/", nruns)
        Y_bar = gen_Y_bars(cdf_Ybar[0], cdf_Ybar[1], N)
        res[run] = calc_quad_tree_ev(x0, Y_bar, N, E=70)
        #print()
    
    print()
    print("Overall EV = ", np.mean(res))
    print("BS Price for assumptions = ", bs_call(80.99, 70, 43/252, 0.0343, 0.234, 0))
    return 0

if __name__ == "__main__":
    main()
