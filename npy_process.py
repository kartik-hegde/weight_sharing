import numpy as np
from math_helpers import *

#Top level function to convert the npy to weight sharing
def convert_npy_kmeans(x, scale_factor, data_16bit, isBias):
	key_list = x.keys()
	return_dict = {}

	for i in key_list:
		print "Converting Layer: " + str(i)
		if 'conv' in i:
			temp_list = convert_kmeans_conv(x[i], scale_factor, data_16bit, isBias)
		elif 'fc' in i:
			temp_list = convert_kmeans_fc(x[i], scale_factor, data_16bit, isBias)
		else:
			temp_list = x[i]
		return_dict[i] = temp_list
	print "Done!"
	return return_dict

#def convert_npy_normal(x, Nconv, Nfc):
def convert_npy_normal(x, scale_factor, isBias):
	key_list = x.keys()
	return_dict = {}

	for i in key_list:
		print "Converting Layer: " + str(i)
		if 'conv' in i:
			temp_list = convert_normal_conv(x[i], scale_factor, isBias)
		elif 'fc' in i:
			temp_list = convert_normal_fc(x[i], scale_factor, isBias)
		else:
			temp_list = x[i]
		return_dict[i] = temp_list
	print "Done!"
	return return_dict


#Top level function to convert the npy to 8b
def convert_npy_8b(x, isBias):
	key_list = x.keys()
	return_dict = {}

	for i in key_list:
		print "Converting Layer: " + str(i)
		if 'conv' in i:
			temp_list = convert_8b_conv(x[i], isBias)
		elif 'fc' in i:
			temp_list = convert_8b_fc(x[i])
		else:
			temp_list = x[i]
		return_dict[i] = temp_list
	print "Done!"
	return return_dict

#Top level function to convert the npy to 16b
def convert_npy_16b(x, isBias):
	key_list = x.keys()
	return_dict = {}

	for i in key_list:
		print "Converting Layer: " + str(i)
		if 'conv' in i:
			temp_list = []
			#for j in range(int(isBias)+1): #Weights and biases
			if isBias:
				for j in range(2): #Weights and biases
					temp_list.append(np.asarray(x[i][j], dtype=np.float16))
			else:
				for j in range(1): #Weights
					temp_list.append(np.asarray(x[i][j], dtype=np.float16))
			return_dict[i] = temp_list 
		else:
			return_dict[i] = x[i]
	print "Done!"
	return return_dict

#Convert 1 conv layer to 8b
def convert_8b_conv(x, isBias):
	x0 = x[0] #Filters
	
	return_list = []

	p,q,r,s = x0.shape
	for i in range(p):
		for j in range(q):
			for k in range(r):
				for l in range(s):
					x0[i][j][k][l] = byte_to_32b(byte_fixed(x0[i][j][k][l]))
	return_list.append(x0)
	if isBias:
		x1 = x[1] #Biases
		t = x1.shape[0]
		for i in range(t):
			x1[i] = byte_to_32b(byte_fixed(x1[i]))

		return_list.append(x1)

	return return_list

#Convert 1 fc layer to 8b
def convert_8b_fc(x):
	x0 = x[0] #Filters
	x1 = x[1] #Biases
	return_list = []
	p,q = x0.shape
	for i in range(p):
		for j in range(q):
			x0[i][j] = byte_to_32b(byte_fixed(x0[i][j]))
	return_list.append(x0)

	t = x1.shape[0]
	for i in range(t):
		x1[i] = byte_to_32b(byte_fixed(x1[i]))
	return_list.append(x1)

	return return_list

#Convert 1 conv layer to weight sharing with kmeans
def convert_kmeans_conv(x, scale_factor, data_16bit, isBias):
	x0 = x[0] #Filters
	return_list = []

	p,q,r,s = x0.shape
	#Chop off requested number of bits from the number of Unique weights
	num_centroids = 2 ** (int(np.log2(4*p*q*r/3)) - scale_factor)
	for l in range(s):
		temp_list=[]
		#Read the values into a list
		for i in range(p):
			for j in range(q):
				for k in range(r):
					#Convert to 32bit
					if data_16bit:
						temp_data = np.float32(x0[i][j][k][l])
					else:
						temp_data = x0[i][j][k][l]

					temp_list.append(temp_data)

		temp_list = np.asarray(temp_list, dtype=np.float32)
		temp_list_centroids, templist_index = kmeans_data(temp_list, num_centroids)

		#Replace the values in temp list
		for i,item in enumerate(temp_list):
			temp_list[i] = temp_list_centroids[templist_index[i]]
		#Write Back to the original ndarray
		for i in range(p):
			for j in range(q):
				for k in range(r):
					x0[i][j][k][l] = temp_list[i*q*r+j*r+k]

	return_list.append(x0)

	if isBias:
		#Do nothing for biases
		x1 = x[1] #Biases
		return_list.append(x1)

	return return_list

#Convert 1 fc layer to weight sharing with kmeans
def convert_kmeans_fc(x, scale_factor, data_16bit, isBias):
	x0 = x[0] #Filters
	return_list = []

	p,q = x0.shape
	#Chop off requested number of bits from the number of Unique weights
	num_centroids = 2 ** (int(np.log2(q)) + 1 - scale_factor)
	temp_list=[]
	#Read the values into a list
	for i in range(p):
		#Convert to 32bit
		if data_16bit:
			temp_list = np.asarray(x0[i], dtype=np.float32)
		else:
			temp_list = x0[i]
		
		temp_list_centroids, templist_index = kmeans_data(temp_list, num_centroids )

		#Replace the values in temp list
		for j,item in enumerate(temp_list):
			temp_list[j] = temp_list_centroids[templist_index[j]]
		#Write Back to the original ndarray
		x0[i] = temp_list
	return_list.append(x0)

	if isBias:
		#Do nothing for biases
		x1 = x[1] #Biases
		return_list.append(x1)

	return return_list

#Convert 1 conv layer to weight sharing with normal distribution
def convert_normal_conv(x, scale_factor, isBias):
	x0 = x[0] #Filters
	return_list = []

	p,q,r,s = x0.shape
	for l in range(s):
		temp_list=[]
		#Read the values into a list
		for i in range(p):
			for j in range(q):
				for k in range(r):
					temp_list.append(x0[i][j][k][l])

		temp_list_normal = normal_data(temp_list, scale_factor)
		#Write Back to the original ndarray
		for i in range(p):
			for j in range(q):
				for k in range(r):
					x0[i][j][k][l] = temp_list_normal[i*q*r+j*r+k]

	return_list.append(x0)
	if isBias:
		#Do nothing for biases
		x1 = x[1] #Biases
		return_list.append(x1)

	return return_list

#Convert 1 fc layer to weight sharing with normal distribution
def convert_normal_fc(x, scale_factor, isBias):
	x0 = x[0] #Filters
	return_list = []

	p,q = x0.shape
	temp_list=[]
	#Read the values into a list
	for i in range(p):
		temp_list = x0[i]

		temp_list_normal = normal_data(temp_list, N)
		#Write Back to the original ndarray
		x0[i] = temp_list_normal
	return_list.append(x0)

	if isBias:
		x1 = x[1] #Biases
		#Do nothing for biases
		return_list.append(x1)

	return return_list


#return a fixed point number Q07
def byte_fixed(x):
	return int(x * 2**8)

#Return a 32b Float
def byte_to_32b(x):
	return (x*1.0) / 2**8

#return a fixed point number Q07 for the entire list
def byte_fixed_list(x):
	converted=[]
	for i in x:
		converted.append(int(i * 2**8))
	return converted

#Return a 32b Float for the entire list
def byte_to_32b_list(x):
        converted=[]
	for i in x:
	 converted.append((i*1.0) / 2**8)
	return converted

#Flatten the array in terms of RSC
def flatten_rsc(x, K):
	p,q,r,s = x.shape
	flat=[]
	for i in range(p):
		for j in range(q):
			for k in range(r):
				flat.append(x[i][j][k][K])
	return flat

#Flatten the Array in terms of RSK
def flatten_rsk(x):
	p,q,r,s = x.shape
	flat=[]
	for i in range(p):
		for j in range(q):
			for k in range(r):
				flat.append(x[i][j][0][k])
	return flat

#Return a bucket of repetitions of values
def get_count(x): #receive a flat list
	count_bin= np.zeros((5000,), dtype=np.int)
	used=[]
	for i in x:
		if(i not in used and i != 0):
			count = x.count(i)
			count_bin[count]=count_bin[count]+1
			used.append(i)
	return count_bin
