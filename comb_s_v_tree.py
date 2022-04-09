import numpy as np

def calc_vn(y, v, k, theta, v_pos, dt, omega):
    return(v+k*(theta-v_pos)*dt+y*omega*np.sqrt(v_pos*dt))

def calc_zn(y, v, z, r, v_pos, dt):
    return(z+(r-0.5*v)*dt+y*np.sqrt(v_pos*dt))

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
                # i_1 and i_2 are coded for 0 = -1
                # So v0 means y_1 = -1 and v1 means y_1 = 1
                v_pos = max(grid_v[k,j], 0)
                z0 = calc_zn(-1, grid_v[k,j], grid_z[k,i], r, v_pos, dt)
                z1 = calc_zn(1, grid_v[k,j], grid_z[k,i], r, v_pos, dt)
                v0 = calc_vn(-1, grid_v[k,j], K, theta, v_pos, dt, omega)
                v1 = calc_vn(1, grid_v[k,j], K, theta, v_pos, dt, omega)
                if(z1 < z0):
                    temp = z0
                    z0 = z1
                    z1 = temp
                if(v1 < v0):
                    temp = v0
                    v0 = v1
                    v1 = temp
                # Find the successors values
                #indices for the lower left corner
                z0_i = int(np.floor((z0-grid_z[k+1,0])/dz[k+1]))
                z1_i = int(np.floor((z1-grid_z[k+1,0])/dz[k+1]))
                v0_i = int(np.floor((v0-grid_v[k+1,0])/dv[k+1]))
                v1_i = int(np.floor((v1-grid_v[k+1,0])/dv[k+1]))
                z1_i_1 = min(z1_i+1, mz-1)
                v1_i_1 = min(v1_i+1, mv-1)
                
                def tilde_val(x, x0s, x1s): return (x-x0s)/(x1s-x0s)
                
                def c00(x_tilde, y_tilde): return (1-x_tilde)*(1-y_tilde)
                def c10(x_tilde, y_tilde): return x_tilde*(1-y_tilde)
                def c01(x_tilde, y_tilde): return (1-x_tilde)*y_tilde
                def c11(x_tilde, y_tilde): return x_tilde*y_tilde

                z0_til = tilde_val(z0, grid_z[k+1,z0_i], grid_z[k+1,z0_i+1])
                # If at the boundary use 1
                z1_til = 1
                if z1_i != z1_i_1:
                   z1_til = tilde_val(z1, grid_z[k+1,z1_i], grid_z[k+1,z1_i_1])
                v0_til = tilde_val(v0, grid_v[k+1,v0_i], grid_v[k+1,v0_i+1])
                # If at the boundary use 1
                v1_til = 1
                if v1_i != v1_i_1:
                    v1_til = tilde_val(v1, grid_v[k+1,v1_i], grid_v[k+1,v1_i_1])

                # i_1 and i_2 are coded for 0 = -1
                # So q0101 means i_1 = -1, i_2 = 1, i_3 = 0, i_4 = 1
                q0000 = 0.25*(1+rho)*c00(v0_til, z0_til)*grid_s[k+1,z0_i,v0_i]
                q0001 = 0.25*(1+rho)*c01(v0_til, z0_til)*grid_s[k+1,z0_i,v0_i+1]
                q0010 = 0.25*(1+rho)*c10(v0_til, z0_til)*grid_s[k+1,z0_i+1,v0_i]
                q0011 = 0.25*(1+rho)*c11(v0_til, z0_til)*grid_s[k+1,z0_i+1,v0_i+1]

                q0100 = 0.25*(1-rho)*c00(v0_til, z1_til)*grid_s[k+1,z0_i,v1_i]
                q0101 = 0.25*(1-rho)*c01(v0_til, z1_til)*grid_s[k+1,z0_i,v1_i_1]
                q0110 = 0.25*(1-rho)*c10(v0_til, z1_til)*grid_s[k+1,z0_i+1,v1_i]
                q0111 = 0.25*(1-rho)*c11(v0_til, z1_til)*grid_s[k+1,z0_i+1,v1_i_1]

                q1000 = 0.25*(1-rho)*c00(v1_til, z0_til)*grid_s[k+1,z1_i,v0_i]
                q1001 = 0.25*(1-rho)*c01(v1_til, z0_til)*grid_s[k+1,z1_i,v0_i+1]
                q1010 = 0.25*(1-rho)*c10(v1_til, z0_til)*grid_s[k+1,z1_i_1,v0_i]
                q1011 = 0.25*(1-rho)*c11(v1_til, z0_til)*grid_s[k+1,z1_i_1,v0_i+1]

                q1100 = 0.25*(1+rho)*c00(v1_til, z1_til)*grid_s[k+1,z1_i,v1_i]
                q1101 = 0.25*(1+rho)*c01(v1_til, z1_til)*grid_s[k+1,z1_i,v1_i_1]
                q1110 = 0.25*(1+rho)*c10(v1_til, z1_til)*grid_s[k+1,z1_i_1,v1_i]
                q1111 = 0.25*(1+rho)*c11(v1_til, z1_til)*grid_s[k+1,z1_i_1,v1_i_1]

                grid_s[k,i,j] = q0000+q0001+q0010+q0011+q0100+q0101+q0110+q0111+q1000+q1001+q1010+q1011+q1100+q1101+q1110+q1111

    print(grid_s[0][0][0])

if __name__ == "__main__":
    #calc_sv_tree(n, mz, mv, Z0, V0)
    calc_sv_tree(n=25, mz=125, mv=6, Z0=np.log(10), V0=0.25**2)
    # Run result: 0.9819760855018189
    #calc_sv_tree(n=71, mz=1000, mv=48, Z0=np.log(10), V0=0.25**2)
    # Run result: 0.6123673190802092