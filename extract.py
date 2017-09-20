import numpy as np
from npy_process import *

data = np.load("data_kmeans.npy").item()

conv3 = data['conv3']
conv3 = conv3[0]
conv3_f16 = np.asarray(conv3, dtype=np.float16)



conv3_flat = flatten_rsc(conv3)
conv3_f16_flat = flatten_rsc(conv3_f16)
conv3_flat_rsk = flatten_rsk(conv3)
conv3_f16_flat_rsk = flatten_rsk(conv3_f16)

conv3_byte_flat = byte_fixed_list(conv3_flat)
conv3_byte_flat_rsk = byte_fixed_list(conv3_flat_rsk)



print get_count(conv3_flat)
print get_count(conv3_f16_flat)
print get_count(conv3_flat_rsk)
print get_count(conv3_f16_flat_rsk)
print get_count(conv3_byte_flat)
print get_count(conv3_byte_flat_rsk)
#print conv3_flat
#print byte_to_32b_list(conv3_byte_flat)
