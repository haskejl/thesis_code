import numpy as np

def calc_vn(y, v, k, theta, v_pos, dt, omega):
    return(v+k*(theta-v_pos)*dt+y*omega*np.sqrt(v_pos*dt))

def calc_zn(y, v, z, r, v_pos, dt):
    return(z+(r-0.5*v)*dt+y*np.sqrt(v_pos*dt))

def bilinear_interp(X_arr, Y_arr, x_ind, y_ind, x, y, f):
    x_ind = int(x_ind)
    y_ind = int(y_ind)
    if x_ind >= len(X_arr)-1:
        if y_ind >= len(Y_arr)-1:
            # Both on upper edge, no interpolation required
            return f[x_ind, y_ind]
        # X on upper edge, only interpolate Y
        Cy = (y-Y_arr[y_ind])/(Y_arr[y_ind+1]-Y_arr[y_ind])
        assert Cy >= 0 and Cy <= 1, "Linear Cy doesn't meet probability reqs, Cy =" + str(Cy)
        return (1-Cy) * f[x_ind, y_ind] + Cy * f[x_ind, y_ind+1]

    if y_ind >= len(Y_arr)-1:
        # Y on upper edge, only interpolate X
        Cx = (x-X_arr[x_ind])/(X_arr[x_ind+1]-X_arr[x_ind])
        assert Cx >= 0 and Cx <= 1, "Linear Cx doesn't meet probability reqs, Cx =" + str(Cx)
        return (1-Cx) * f[x_ind, y_ind] + Cx * f[x_ind+1, y_ind]
    # Neither on upper edge, bilinear interpolation for both
    Cx = (x-X_arr[x_ind])/(X_arr[x_ind+1]-X_arr[x_ind])
    Cy = (y-Y_arr[y_ind])/(Y_arr[y_ind+1]-Y_arr[y_ind])
    assert Cy >= 0 and Cy <= 1, "Bilinear Cy doesn't meet probability reqs, Cy =" + str(Cy)
    assert Cx >= 0 and Cx <= 1, "Bilinear Cx doesn't meet probability reqs, Cx =" + str(Cx)

    # Floating point error creeps in here
    return (1-Cy)*((1-Cx)*f[x_ind,y_ind] + Cx*f[x_ind+1, y_ind]) + Cy*((1-Cx)*f[x_ind, y_ind+1] + Cx*f[x_ind+1, y_ind+1])

def calc_sv_tree(n, mz, mv, Z0, V0):
    T = 0.25
    dt = T/n
    # Lk 14
    K = 5
    theta = 0.16
    omega = 0.9
    rho = 0.1
    r = 0.1
    E = 10

    v_tilde_max = np.zeros(n)
    v_tilde_min = np.zeros(n)
    z_tilde_max = np.zeros(n)
    z_tilde_min = np.zeros(n)

    z_tilde_min[0] = z_tilde_max[0] = Z0
    v_tilde_min[0] = v_tilde_max[0] = V0
    # Find the ranges for v and z of the grid
    for k in range(1, n):
        v_pos_max = max(v_tilde_max[k-1], 0)
        v_pos_min = max(v_tilde_min[k-1], 0)
        v_max_successor1 = calc_vn(1, v_tilde_max[k-1], K, theta, v_pos_max, dt, omega)
        v_max_successor2 = calc_vn(-1, v_tilde_max[k-1], K, theta, v_pos_max, dt, omega)
        v_min_successor1 = calc_vn(1, v_tilde_min[k-1], K, theta, v_pos_min, dt, omega)
        v_min_successor2 = calc_vn(-1, v_tilde_min[k-1], K, theta, v_pos_min, dt, omega)

        z_max_successor1 = calc_zn(1, v_tilde_max[k-1], z_tilde_max[k-1], r, v_pos_max, dt)
        z_max_successor2 = calc_zn(-1, v_tilde_max[k-1], z_tilde_max[k-1], r, v_pos_max, dt)
        z_max_successor3 = calc_zn(1, v_tilde_min[k-1], z_tilde_max[k-1], r, v_pos_min, dt)
        z_max_successor4 = calc_zn(-1, v_tilde_min[k-1], z_tilde_max[k-1], r, v_pos_min, dt)
        z_min_successor1 = calc_zn(1, v_tilde_max[k-1], z_tilde_min[k-1], r, v_pos_max, dt)
        z_min_successor2 = calc_zn(-1, v_tilde_max[k-1], z_tilde_min[k-1], r, v_pos_max, dt)
        z_min_successor3 = calc_zn(1, v_tilde_min[k-1], z_tilde_min[k-1], r, v_pos_min, dt)
        z_min_successor4 = calc_zn(-1, v_tilde_min[k-1], z_tilde_min[k-1], r, v_pos_min, dt)
        
        v_tilde_max[k] = max(v_max_successor1, v_max_successor2)
        v_tilde_min[k] = min(v_min_successor1, v_min_successor2)
        z_tilde_max[k] = max(z_max_successor1, z_max_successor2, z_max_successor3, z_max_successor4)
        z_tilde_min[k] = min(z_min_successor1, z_min_successor2, z_min_successor3, z_min_successor4)
    del(v_pos_max)
    del(v_pos_min)
    
    dz = (z_tilde_max-z_tilde_min)/(mz-1)
    dv = (v_tilde_max-v_tilde_min)/(mv-1)
    
    # Setup the grids
    grid_z = np.linspace(z_tilde_min, z_tilde_max, mz, axis=-1)
    grid_v = np.linspace(v_tilde_min, v_tilde_max, mv, axis=-1)
    grid_s = np.zeros((n, mz, mv))
    del(z_tilde_min)
    del(z_tilde_max)
    del(v_tilde_min)
    del(v_tilde_max)
    
    # Calculate the final prices
    grid_s[n-1,:,0] = np.maximum(E-np.exp(grid_z[n-1,:]), 0)
    
    # It seems to be that all values of v should have the same price for a given z
    #  fill them in here
    for i in range(1,len(grid_s[n-1,0])):
        grid_s[n-1,:,i] = grid_s[n-1,:,0]

    Y_12 = np.array([-1,1])
    for k in range(n-2, -1, -1):
        # For each node in grid_s at time k
        for i in range(0, len(grid_z[k])):
            for j in range(0, len(grid_v[k])):
                # Calculate the successors
                v_pos = max(grid_v[k,j], 0)
                z_k1 = calc_zn(Y_12, grid_v[k,j], grid_z[k,i], r, v_pos, dt)
                v_k1 = calc_vn(Y_12, grid_v[k,j], K, theta, v_pos, dt, omega)

                assert z_k1[0] < z_k1[1], "z_k1[0] > z_k1[1], values:" + str(z_k1)
                assert v_k1[0] < v_k1[1], "v_k1[0] > v_k1[1], values:" + str(v_k1)

                # Find point to the lower left of the successor
                z_ind = np.floor((z_k1-grid_z[k+1,0])/dz[k+1]).astype(int)
                v_ind = np.floor((v_k1-grid_v[k+1,0])/dv[k+1]).astype(int)

                interp_vals = np.zeros(4)
                grid_s[k,i,j]
                i_ind = 0
                for ind in range(0,2):
                    for jnd in range(0,2):
                        interp_vals[i_ind] = bilinear_interp(grid_z[k+1], grid_v[k+1], z_ind[ind], v_ind[jnd], z_k1[ind], v_k1[jnd], grid_s[k+1])
                        i_ind += 1
                grid_s[k,i,j] = np.exp(-r*dt)*0.25 * ((1+rho)*(interp_vals[0]+interp_vals[3]) + (1-rho)*(interp_vals[1]+interp_vals[2]))

    print(np.round(grid_s[0][0][0],4))

if __name__ == "__main__":
    #calc_sv_tree(n, mz, mv, Z0, V0)
    calc_sv_tree(n=25, mz=125, mv=6, Z0=np.log(10), V0=0.25**2)
    # Run result: 0.4825
    calc_sv_tree(n=71, mz=1000, mv=48, Z0=np.log(10), V0=0.25**2)
    # Run result: 0.4954