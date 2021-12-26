import numpy as np

class QuadTreeNode:
    # Class is similar to a skip list
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

    def print_me(self):
        print("Me: ", self.x)

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
    if(cdf[0] == 0):
        start = 1
    for i in range(1,len(cdf)):
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
    return (use_cdf(cdf, Y_primes), cdf)

def calc_cdf(X):
    #n is selected from p.23 (1000)
    n = 100
    #K is the length of the historical data (in days for daily data)
    K = len(X)
    cdf_out = np.zeros(n)
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
    print(Y_prime_out)
    print(cdf_out)
    return(cdf_out, Y_prime_out)
    #Y_bar = np.zeros(N)
    #for i in range(0,N):
    #    Y_bar[i] = use_cdf(cdf_out, Y_prime_out, n)
    #return Y_bar

def gen_Y_bars(cdf, Y_prime, N):
    Y_bar = np.zeros(N)
    for i in range(0,N):
        Y_bar[i] = use_cdf(cdf, Y_prime)
    return Y_bar

def payoff_func(x):
    return max(x-70,0)

def calc_quad_tree_ev(x0, Y_bar, N):
    T = 1
    p=0.135
    # r value is from p. 25
    r = 0.0343
    dt = T/N
    print("Calculating the tree...")
    # Set the base node for the tree
    top_node = bottom_node = base_node = QuadTreeNode(x0, 1)
    sig = calc_sigma(Y_bar)
    add_pt = (r-sig**2/2)*dt
    mul_pt = sig*np.sqrt(dt)

    for i in range(0,N):
        #print("Running step: ", i)
        # Must be a ceiling function or j may end up below the point
        j_upp = int(np.ceil((top_node.x - add_pt[i])/mul_pt[i]))
        j_downn = int(np.ceil((bottom_node.x - add_pt[i])/mul_pt[i]))
        '''j_upp = int(np.ceil(top_node.x/mul_pt))
        j_downn = int(np.ceil(bottom_node.x/mul_pt))'''
        j = range(j_upp+1, j_downn-3,-1)
        nodes = {j[0]: QuadTreeNode(j*mul_pt[i]+add_pt[i])}
        for k in j:
            nodes = nodes | {k: QuadTreeNode(k*mul_pt[i]+add_pt[i])}
        # Since the nodes are shared, set up the linked list here
        for k in j[0:(len(j)-1)]:
            nodes[k].under_me = nodes[k-1]
        

        curr_node = top_node
        last_downn = None
        while(curr_node != None):
            # TODO: What about the drift term? It is used in some places in the paper
            #     but not in others
            node_j = int(np.ceil((curr_node.x-add_pt[i])/mul_pt[i]))
            #node_j = int(np.ceil(curr_node.x/mul_pt))
            d1 = 0
            d2 = 0
            if(i >= 0):
                d1 = curr_node.x - nodes[node_j].x#node_j*mul_pt[i]+add_pt[i]
                d2 = curr_node.x - nodes[node_j-1].x#(node_j-1)*mul_pt[i]+add_pt[i]
                if d1 > 0 or d2 < 0:
                    print("d1 = ", d1, " d2 = ", d2)
                    print("x: = ", curr_node.x, " up = ", nodes[node_j], " down = ", nodes[node_j-1])
                assert d1 <= 0, "d1 should be <= 0 at i==0"
                assert d2 >= 0, "d2 should be >= 0 at i==0"
            else:
                d1 = curr_node.x-add_pt[i-1] - node_j*mul_pt[i]
                d2 = curr_node.x-add_pt[i-1] - (node_j-1)*mul_pt[i]
                if d1 > 0 or d2 < 0: print("d1 = ", d1, " d2 = ", d2)
                assert d1 <= 0, "d1 should be <= 0 at i>0"
                assert d2 >= 0, "d2 should be >= 0 at i>0"
            
            #if(nodes[node_j].x - curr_node.x < curr_node.x - nodes[node_j-1].x):
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
                if((p1 < 0 or p1 > 1) or (p2 < 0 or p2 > 1) or (p3 < 0 or p3 > 1)):
                    print("1x = ", curr_node.x, " j = ", node_j, " sig*dt = ", mul_pt, " q = ", q)
                    print(p1, " ", p2, " ", p3, " ", p4, "   ", p1+p2+p3+p4)
                    print()
                    assert p1 > 0, "p1 should be > 0 for -d1 < d2"
                    assert p2 > 0, "p2 should be > 0 for -d1 < d2"
                    assert p3 > 0, "p3 should be > 0 for -d1 < d2"
                    assert p1 < 1, "p1 should be < 1 for -d1 < d2"
                    assert p2 < 1, "p2 should be < 1 for -d1 < d2"
                    assert p3 < 1, "p3 should be < 1 for -d1 < d2"
            else:
                q = d2/mul_pt[i]
                p1 = p
                p2 = 0.5*(1+q+q**2)-3*p
                p3 = 3*p-q**2
                p4 = 0.5*(1-q+q**2)-p
                if((p2 < 0 or p2 > 1) or (p3 < 0 or p3 > 1) or (p4 < 0 or p4 > 1)):
                    print("2x = ", curr_node.x, " j = ", node_j, " sig*dt = ", mul_pt, " q = ", q)
                    print(p1, " ", p2, " ", p3, " ", p4, "   ", p1+p2+p3+p4)
                    print()
                    assert p4 > 0, "p4 should be > 0 for -d1 > d2"
                    assert p2 > 0, "p2 should be > 0 for -d1 > d2"
                    assert p3 > 0, "p3 should be > 0 for -d1 > d2"
                    assert p4 < 1, "p4 should be < 1 for -d1 > d2"
                    assert p2 < 1, "p2 should be < 1 for -d1 > d2"
                    assert p3 < 1, "p3 should be < 1 for -d1 > d2"

            curr_node.upp = nodes[node_j+1]
            curr_node.upp.probability = curr_node.upp.probability + p1*curr_node.probability
            curr_node.up = nodes[node_j]
            curr_node.up.probability = curr_node.up.probability + p2*curr_node.probability
            curr_node.down = nodes[node_j-1]
            curr_node.down.probability = curr_node.down.probability + p3*curr_node.probability
            curr_node.downn = nodes[node_j-2]
            curr_node.downn.probability = curr_node.downn.probability + p4*curr_node.probability
            
            curr_node = curr_node.under_me
        top_node = top_node.upp
        bottom_node = bottom_node.downn
    
    # Calculate the ev of the payoff of the final row of options
    print("Calculating EV...")
    curr_node = top_node
    ret_val = 0
    tot_prob = 0
    while(curr_node != None):
        #print("x = ", curr_node.x)
        #print("p = ", curr_node.probability)
        tot_prob = tot_prob + curr_node.probability
        ret_val = ret_val + payoff_func(np.exp(curr_node.x))*curr_node.probability
        #print("res = ", res[o])
        curr_node = curr_node.under_me
    print("EV for run = ", ret_val)
    print("Total Prob for run = ", tot_prob)
    '''print()
    node = base_node
    while(node != None):
        string = format(node.x, ".2f")
        list_node = node.under_me
        while(list_node != None):
            string = format(list_node.x, ".2f") + " " + string
            list_node = list_node.under_me
        node = node.upp
        print(string)'''
    return(ret_val)

def main():
    #Previous year's IBM close dates (Jul 19, 2004 to July 18, 2005)
    data = open("./data/10days_IBM.csv")
    lines = data.readlines()
    data.close()

    X_vals = np.zeros(len(lines)-1)
    for i in range(0,len(lines)-1):
        # TODO: Should this have the log taken since data is transformed later
        X_vals[i] = np.log(float(lines[i+1].split(",")[4].strip()))
    nruns = 100
    N = 100
    x0 = X_vals[-1] #could also set to 80.99 the open price for July 19, 2005
    res = np.zeros(nruns)
    print("Calculating cdf...")
    cdf_Ybar = calc_cdf(X_vals)

    for o in range(0,nruns):
        #print("Run #", o)
        #print("Calculating Y bar values...")
        Y_bar = gen_Y_bars(cdf_Ybar[0], cdf_Ybar[1], N)
        
        res[o] = calc_quad_tree_ev(x0, Y_bar, N)

        #print("Curr est EV = ", np.mean(res[0:(o+1)]))
    
    print()
    print("Overall EV = ", np.mean(res))
    return 0

if __name__ == "__main__":
    main()
