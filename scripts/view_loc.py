import matplotlib.pyplot as plt
import numpy as np

def view_localization_estimation(localization_file):
    data = np.load(localization_file, allow_pickle=True)
    T_w_velo_est = data['T_w_velo_est']
    
    plt.plot(T_w_velo_est[:, 0, 3], T_w_velo_est[:, 1, 3], 'r', label='Estimated Trajectory')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Localization Estimation')
    plt.legend()
    plt.show()

localization_file = 'localization_2023-11-30_23-06-03.npz'
view_localization_estimation(localization_file)
