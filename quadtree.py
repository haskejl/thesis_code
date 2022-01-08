import numpy as np

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

#sigma as defined on page 23
def calc_sigma(y):
    return np.exp(-abs(y))

def payoff_func(x, E, r, T):
    return max(x-E,0)*np.exp(-r*T)

def calc_quad_tree_ev(x0, Y_bar, N, E):
    # Time is 43 trading days from Jul. 19 to Sep. 16 since sigma_sqility was calculated based on days
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
            nodes.update({k: QuadTreeNode(k*mul_pt[i]+add_pt[i])})
            #nodes = nodes | {k: QuadTreeNode(k*mul_pt[i]+add_pt[i])} # Requires Python 3.9+
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
        
        curr_node = top_node
        last_node = top_node
        assert top_node.probability != 0, "Error: the top node probability is zero"
        while(curr_node != None):
            if(curr_node.probability == 0):
                last_node.under_me = curr_node.under_me
            else:
                last_node = curr_node
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
    Y_bar = np.random.normal(size=1)
    N = 10
    calc_quad_tree_ev(x0, Y_bar, N, E=70)

if __name__ == "__main__":
    main()
