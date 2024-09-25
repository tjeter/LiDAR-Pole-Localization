import matplotlib.pyplot as plt
import numpy as np

def view_local_maps(localmapfile):
    with np.load(localmapfile, allow_pickle=True) as data:
        maps = data['maps']
        for i, localmap in enumerate(maps):
            poleparams = localmap['poleparams']
            plt.scatter(poleparams[:, 0], poleparams[:, 1], s=5, label=f'Local Map {i}', alpha=0.7)

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Local Pole Maps')
        plt.show()

localmapfile = 'localmaps_3.npz'
view_local_maps(localmapfile)
