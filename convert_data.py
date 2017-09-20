import numpy as np
from npy_process import *
NUM_WEIGHTS_CONV = 256
NUM_WEIGHTS_FC = 128
data = np.load("bvlc_alexnet.npy").item()

#data_8bit = convert_npy_8b(data)
#np.save("data_8bit.npy", data_8bit)

#data_16bit = convert_npy_16b(data)
#np.save("data_16bit.npy", data_16bit)

data_kmeans = convert_npy_kmeans(data)
data_normal = convert_npy_normal(data, NUM_WEIGHTS_CONV, NUM_WEIGHTS_FC)
np.save("data_kmeans.npy", data_kmeans)
np.save("data_normal.npy", data_normal)

