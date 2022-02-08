import numpy as np

def calc_vn(y, v, z, dtn):
    return(v+k*(theta-v_pos)*dtn+y*omega*np.sqrt(v_pos*dt))

def calc_zn(y, v, z, dtn):
    return(z+(r-0.5*v)*dtn+y*np.sqrt(v_pos*dtn))

def get_Yn1_Yn2(rho):
    ij_vals = np.array([-1, 1])
    Q_tilde = np.array([0, 0, 0, 0])
    cdf_Q = 0
    for i in ij_vals:
        for j in ij_vals:
            Q_tilde[i, j] = 0.25*(1+i*j*rho)
            cdf_Q = cdf_Q + Q_tilde[i, j]
    assert cdf_Q == 1, "CDF for Y1 and Y2 = ", str(cdf_Q), " not 1"

    rn = np.random.uniform()
    for i in range(0, 4):
        if(rn <= cdf_Q): # TODO: double check the returns are under the right i values
            if(i == 0):
                return 1, 1
            if(i == 1):
                return 1, -1
            if(i == 2):
                return -1, 1
            if(i == 3):
                return -1, -1

def calc_sv_tree(n, mz, mv, V0, Z0):
    zmax = np.zeros(k)
    zmin = np.zeros(k)
    vmax = np.zeros(k)
    vmin = np.zeros(k)

    dtn = T/n
    rho = something
    for k in range(0, n):
        Yn1, Yn2 = get_Yn1_Yn2(rho)
        v_tilde[n+1] = xsy*(calc_vn(Yn1, v_tilde[n], z_tilde[n]))
        z_tilde[n+1] = ysy*(calc_zn(Yn2, v_tilde[n], z_tilde[n]))
        zmax = 1
        zmin = 1
        vmax = 1
        vmin = 1
    
    dz = (zmax-zmin)/mz
    dv = (vmax-vmin)/mv

