import numpy as np

def calc_vn(y, v, z, k, theta, v_pos, dtn, omega):
    return(v+k*(theta-v_pos)*dtn+y*omega*np.sqrt(v_pos*dtn))

def calc_zn(y, v, z, r, dtn, v_pos):
    return(z+(r-0.5*v)*dtn+y*np.sqrt(v_pos*dtn))

def get_Yn1_Yn2(rho):
    ij_vals = np.array([1, -1])
    cdf_Q = np.zeros(4)
    loop_ctr = 0
    for i in ij_vals:
        for j in ij_vals:
            cdf_Q[loop_ctr] = cdf_Q[loop_ctr-1] + 0.25*(1+i*j*rho)
            loop_ctr = loop_ctr + 1
    assert cdf_Q[3] == 1, "CDF for Y1 and Y2 = " + str(cdf_Q) + " not 1"

    rn = np.random.uniform()
    for i in range(0, 4):
        if(rn <= cdf_Q[i]):
            if(i == 0):
                return 1, 1
            if(i == 1):
                return 1, -1
            if(i == 2):
                return -1, 1
            if(i == 3):
                return -1, -1

def calc_sv_tree(n, mz, mv, Z0, V0):
    zmax = np.zeros(n)
    zmin = np.zeros(n)
    vmax = np.zeros(n)
    vmin = np.zeros(n)

    T = 1 # lk. 8
    dtn = T/n
    # Lk 14
    K = 5
    theta = 0.16
    omega = 0.9
    rho = 0.1
    r = 0.1
    E = 10 # arbitrary

    v_tilde_max = np.zeros(n)
    v_tilde_min = np.zeros(n)
    z_tilde_max = np.zeros(n)
    z_tilde_min = np.zeros(n)

    z_tilde_min[0] = z_tilde_max[0] = Z0
    v_tilde_min[0] = v_tilde_max[0] = V0
    # Find the ranges for v and z of the grid
    for k in range(1, n):
        Yn1, Yn2 = get_Yn1_Yn2(rho)
        v_pos_max = np.max(v_tilde_max[k-1], 0)
        v_pos_min = np.max(v_tilde_min[k-1], 0)
        # Processes are supposed to be positive, but go negative sometimes
        v_tilde_max[k] = (calc_vn(1, v_tilde_max[k-1], z_tilde_max[k-1], K, theta, v_pos_max, dtn, omega))
        v_tilde_min[k] = (calc_vn(-1, v_tilde_min[k-1], z_tilde_min[k-1], K, theta, v_pos_min, dtn, omega))
        z_tilde_max[k] = (calc_zn(1, v_tilde_max[k-1], z_tilde_max[k-1], r, dtn, v_pos_max))
        z_tilde_min[k] = (calc_zn(-1, v_tilde_min[k-1], z_tilde_min[k-1], r, dtn, v_pos_min))
    
    dz = (z_tilde_max-z_tilde_min)/mz
    dv = (v_tilde_max-v_tilde_min)/mv
    
    # Setup the grids
    grid_z = np.linspace(z_tilde_min, z_tilde_max, mz, axis=-1)
    grid_v = np.linspace(v_tilde_min, v_tilde_max, mv, axis=-1)
    grid_s = np.zeros((n, mz, mv))
    
    # Calculate the final prices
    grid_s[n-1,:,0] = np.exp(-r*T)*np.maximum(E-np.exp(grid_z[n-1,:]), 0)
    
    # It seems to be that all values of v should have the same price for a given z
    #  fill them in here
    for i in range(1,len(grid_s[n-1,0])):
        grid_s[n-1,:,i] = grid_s[n-1,:,0]

    for k in range(n-2, -1, -1):
        # For each node in grid_s at time k
        for i in range(0, len(grid_z[k])):
            for j in range(0, len(grid_v[k])):
                # Calculate the successor's coordinates
                v_pos = np.max(grid_v[k,j], 0)
                z_new_up = calc_zn(1, grid_v[k,j], grid_z[k,i], r, dtn, v_pos)
                z_new_down = calc_zn(-1, grid_v[k,j], grid_z[k,i], r, dtn, v_pos)
                v_new_up = calc_vn(1, grid_v[k,j], grid_z[k,i], K, theta, v_pos, dtn, omega)
                v_new_down = calc_vn(-1, grid_v[k,j], grid_z[k,i], K, theta, v_pos, dtn, omega)
                # Find the successors values
                #indices for the lower left corner
                z_up_index = np.floor(z_new_up/dz)
                z_down_index = np.floor(z_new_down/dz)
                v_up_index = np.floor(v_new_up/dv)
                v_down_index = np.floor(v_new_down/dv)

                #interpolate
                #s_zup_vup = 

        # Setup the grid of option values for the next iteration
        # TODO: this is test code currently
        grid_s[k,:,0] = np.exp(-r*T)*np.maximum(E-np.exp(grid_z[k,:]), 0)
        for i in range(1, len(grid_s[k,0])):
            grid_s[k,:,i] = grid_s[k,:,0]
    print(grid_s)

