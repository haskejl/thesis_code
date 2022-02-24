import numpy as np
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

T = 42/252 
# r value is from p. 25
r = 0.0343

# Class contains a Quadrinomial tree data structure with a singularly linked list at each time
#  moment. This data structure could be replaced by 2 linked lists in the algorithm to reduce
#  the memory overhead, but provides good functionality for testing and debugging.
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

    def set_successors(self, nodes, node_j):
        self.upp = nodes[node_j+1]
        self.up = nodes[node_j]
        self.down = nodes[node_j-1]
        self.downn = nodes[node_j-2]

    def set_succ_probs(self, p1, p2, p3, p4):
        self.upp.probability = self.upp.probability + p1*self.probability
        self.up.probability = self.up.probability + p2*self.probability
        self.down.probability = self.down.probability + p3*self.probability
        self.downn.probability = self.downn.probability + p4*self.probability

    # Removes any nodes with P(node)=0 under this node in the current time moment
    def remove_p_eq_0_nodes(self):
        curr_node = last_node = self
        assert self.probability != 0, "Error: this node's probability is zero"

        while(curr_node != None):
            if(curr_node.probability == 0):
                last_node.under_me = curr_node.under_me
            else:
                last_node = curr_node
            curr_node = curr_node.under_me

    def print_mini_tree(self):
        print("Me: ", self.x)
        print(self.upp.x, " ", self.up.x, " ", self.down.x, " ", self.downn.x)
    
    def plot_tree(self, depth=30):
        curr_node = self
        top_node = self.upp
        fig, ax = plt.subplots()
        for i in range(0,depth):
            while(curr_node != None):
                x = np.array([i, i+1])
                y = np.array([curr_node.x, curr_node.upp.x])
                ax.plot(x , y, color="lawngreen", linewidth=0.2)
                x = np.array([i, i+1])
                y = np.array([curr_node.x, curr_node.up.x])
                ax.plot(x , y, color="green", linewidth=0.2)
                x = np.array([i, i+1])
                y = np.array([curr_node.x, curr_node.down.x])
                ax.plot(x , y, color="firebrick", linewidth=0.2)
                x = np.array([i, i+1])
                y = np.array([curr_node.x, curr_node.downn.x])
                ax.plot(x , y, color="tomato", linewidth=0.2)
                curr_node = curr_node.under_me
            curr_node = top_node
            top_node = curr_node.upp
        #ax.scatter(x,y)
        fig.savefig("fig.png", dpi=1000)
        fig.show()
#### End class QuadTreeNode

#sigma as defined on page 23
def calc_sigma(y):
    return np.exp(-abs(y))

def payoff_func_call(S, E, r, T):
    return max(S-E,0)*np.exp(-r*T)

def payoff_func_put(S, E, r, T):
    return max(E-S,0)*np.exp(-r*T)

def calc_quad_tree_ev(x0, Y_bar, N, E, payoff_func, p):
    dt = T/N
    
    # Set the base node for the tree
    top_node = bottom_node = base_node = QuadTreeNode(x0, 1)
    sig = calc_sigma(Y_bar)
    add_pt = (r-sig**2/2)*dt
    mul_pt = sig*np.sqrt(dt)

    for i in range(0,N):
        j_uppp = int(np.ceil(top_node.x/mul_pt[i])+2)
        j_downn = int(np.ceil(bottom_node.x/mul_pt[i])-2)

        # Calculate all of the successors with the drift term
        nodes = {}
        nodes.update({j_downn: QuadTreeNode(j_downn*mul_pt[i]+add_pt[i])})
        for k in range(j_downn+1, j_uppp):
            nodes.update({k: QuadTreeNode(k*mul_pt[i]+add_pt[i])})
            # Since the nodes are shared, set up the linked list here
            nodes[k].under_me = nodes[k-1]
        
        curr_node = top_node
        while(curr_node != None):
            node_j = int(np.ceil(curr_node.x/mul_pt[i]))

            d1 = curr_node.x - (node_j*mul_pt[i])
            d2 = curr_node.x - ((node_j-1)*mul_pt[i])
            assert d1 <= 0, "d1 = " + str(d1) + ", but should be <= 0"
            assert d2 >= 0, "d2 = " + str(d2) + ", but should be >= 0"

            p1 = p2 = p3 = p4 = 0
            if(-d1 < d2):
                q = d1/mul_pt[i]
                p4 = p
                p1 = 0.5*(1+q+q**2)-p
                p2 = 3*p-q**2
                p3 = 0.5*(1-q+q**2)-3*p
                assert p1 > 0 and p1 < 1, "p1 = " + str(p1) + " for -d1 < d2"
                assert p2 > 0 and p2 < 1, "p2 = " + str(p2) + " for -d1 < d2"
                assert p3 > 0 and p3 < 1, "p3 = " + str(p3) + " for -d1 < d2"
            else:
                q = d2/mul_pt[i]
                p1 = p
                p2 = 0.5*(1+q+q**2)-3*p
                p3 = 3*p-q**2
                p4 = 0.5*(1-q+q**2)-p
                assert p2 > 0 and p2 < 1, "p2 = " + str(p2) + " for -d1 > d2"
                assert p3 > 0 and p3 < 1, "p3 = " + str(p3) + " for -d1 > d2"
                assert p4 > 0 and p4 < 1, "p4 = " + str(p4) + " for -d1 > d2"
            
            #Set the successor values
            curr_node.set_successors(nodes, node_j)
            # Calculate the conditional probabilities of the successors
            curr_node.set_succ_probs(p1, p2, p3, p4)
            # Go to the next node in this period
            curr_node = curr_node.under_me
        # Remove unneeded nodes
        top_node.remove_p_eq_0_nodes()
        
        # Set the top and bottom nodes for the next period
        top_node = top_node.upp
        bottom_node = bottom_node.downn
    # Move to the highest node of the payoff time moment
    curr_node = top_node

    # Calculate the expected value of the payoff
    expected_val = 0
    while(curr_node != None):
        expected_val = expected_val + payoff_func(np.exp(curr_node.x), E, r, T)*curr_node.probability
        curr_node = curr_node.under_me
    return(expected_val)
