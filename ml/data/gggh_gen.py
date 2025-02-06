import numpy as np

def kallen(a, b, c):
    return (a-(np.sqrt(b)+np.sqrt(c))**2)*(a-(np.sqrt(b)-np.sqrt(c))**2)

#Relation between pt_j, beta and theta_t
def theta_from_beta(beta2, pt_j):
    mh2 = 125.35**2
    s = mh2/(1-beta2)
    pf = np.sqrt(kallen(s,mh2,0))/(2*np.sqrt(s))
    sintheta = pt_j/pf
    return np.arcsin(sintheta)

#Pre-calculated beta2 bounds for a few different pt cuts
#The lower bound in beta2 sets an allowed region for theta_t
#Linearly map a point from the unit square to this constrained physical domain
def unit_square_to_gggh_grid(unit_point, pt_j):
    if pt_j == 10:
        beta2_scale = [0.147332+0.01, 0.991983] #pt_j = 10 GeV (up to 1400 GeV)
    elif pt_j == 20:
        beta2_scale = [0.272228+0.01, 0.991983] #pt_j = 20 GeV (up to 1400 GeV)
    elif pt_j == 30:
        beta2_scale = [0.37762+0.01, 0.991983] #pt_j = 30 GeV (up to 1400 GeV)
    elif pt_j == 50:
        beta2_scale = [0.540675+0.01, 0.991983] #pt_j = 50 GeV (up to 1400 GeV)
    elif pt_j == 100:
        beta2_scale = [0.768192+0.01, 0.991983] #pt_j = 100 GeV (up to 1400 GeV)
    elif pt_j == 300:
        beta2_scale = [0.959793+0.01, 0.997584] #pt_j = 300 GeV (up to 2550 GeV)
    else:
        raise Exception('pt_j must be 10,20,30,50,100 or 300 GeV')
    beta2 = (beta2_scale[1] - beta2_scale[0])*unit_point[0]+beta2_scale[0]
    theta_cut = theta_from_beta(beta2, pt_j)
    theta_t_scale = [theta_cut, np.pi-theta_cut]
    theta_t = (theta_t_scale[1] - theta_t_scale[0])*unit_point[1]+theta_t_scale[0]
    return [beta2, theta_t]

if __name__ == "__main__":
    
    file_path_amp = 'examplex1.f5-amp.f64'
    data_amp = np.fromfile(file_path_amp, dtype=np.float64)
    
    file_path_sm = 'examplex1.f5-amp.f64'
    data_sm = np.fromfile(file_path_sm_train, dtype=np.float64)
    
    file_path_p = 'examplex1.f5-pts.f64'
    data = np.fromfile(file_path_p_train, dtype=np.float64).reshape(-1,2)
    
    pt_j = 30
    mh2 = 125.35**2
    beta2, theta_h = unit_square_to_gggh_grid([data[:,0],data[:,1]], pt_j = pt_j)
    
    s = mh2 / (1 - beta2)
    factor = (s - mh2) / (2 * np.sqrt(s))

    q1 = np.column_stack((np.ones(len(data)), np.zeros(len(data)), np.zeros(len(data)), np.sqrt(s) / 2))
    q2 = np.column_stack((np.ones(len(data)), np.zeros(len(data)), np.zeros(len(data)), -np.sqrt(s) / 2))
    p1 = np.column_stack((np.ones(len(data)), np.zeros(len(data)), -np.sin(theta_h) * factor, -np.cos(theta_h) * factor))
    p2 = np.column_stack(((s + mh2) / (2 * np.sqrt(s)), np.zeros(len(data)), (s - mh2) * np.sin(theta_h) / (2 * np.sqrt(s)), (s - mh2) * np.cos(theta_h) / (2 * np.sqrt(s))))

    p = np.concatenate((q1, q2, p1, p2), axis=-1)
    
    np.save('gggh_dataset.npy', np.concatenate((p, data_sm[:, None], data_amp[:, None]), axis=-1))
