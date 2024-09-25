import matplotlib.pyplot as plt
import numpy as np

def view_global_map(globalmapfile):
    with np.load(globalmapfile, allow_pickle=True) as data:
        polemeans = data['polemeans']
        plt.scatter(polemeans[:, 0], polemeans[:, 1], s=5, c='b', marker='s')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Global Pole Map')
        plt.show()

globalmapfile = 'globalmap_3.npz'
view_global_map(globalmapfile)
