import numpy as np


cameras = {}
n_cameras = 49 # or 64
scan_folder = 'D:\Desktop\my_data'
path_to_dtu_cameras = 'D:\Desktop\my_data'
for i in range(n_cameras):
    filename = '{0}/pos_{1}.txt'.format(path_to_dtu_cameras, '%03d' % (i + 1))
    lines = open(filename).read().splitlines()
    lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
    P = np.asarray(lines).astype(np.float32).squeeze()
    P = np.concatenate([P, [[0, 0, 0, 1]]], 0)
    cameras['world_mat_%d' % i] = P
np.savez('{0}/cameras.npz'.format(scan_folder), **cameras)