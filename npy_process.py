import numpy as np
from math_helpers import *

#Top level function to convert the npy to weight sharing
def convert_npy_kmeans(x):
	key_list = x.keys()
	return_dict = {}

	for i in key_list:
		print "Converting Layer: " + str(i)
		if 'conv' in i:
			temp_list = convert_kmeans_conv(x[i])
		elif 'fc' in i:
			temp_list = convert_kmeans_fc(x[i])
		return_dict[i] = temp_list
	print "Done!"
	return return_dict

#Top level function to convert the npy to 8b
def convert_npy_8b(x):
	key_list = x.keys()
	return_dict = {}

	for i in key_list:
		print "Converting Layer: " + str(i)
		if 'conv' in i:
			temp_list = convert_8b_conv(x[i])
		elif 'fc' in i:
			temp_list = convert_8b_fc(x[i])
		return_dict[i] = temp_list
	print "Done!"
	return return_dict

#Top level function to convert the npy to 16b
def convert_npy_16b(x):
	key_list = x.keys()
	return_dict = {}

	for i in key_list:
		print "Converting Layer: " + str(i)
		temp_list = []
		for j in range(2): #Weights and biases
			temp_list.append(np.asarray(x[i][j], dtype=np.float16))
		return_dict[i] = temp_list 
	print "Done!"
	return return_dict

#Convert 1 conv layer to 8b
def convert_8b_conv(x):
	x0 = x[0] #Filters
	x1 = x[1] #Biases
	return_list = []

	p,q,r,s = x0.shape
	for i in range(p):
		for j in range(q):
			for k in range(r):
				for l in range(s):
					x0[i][j][k][l] = byte_to_32b(byte_fixed(x0[i][j][k][l]))
	return_list.append(x0)
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
def convert_kmeans_conv(x):
	x0 = x[0] #Filters
	x1 = x[1] #Biases
	return_list = []

	p,q,r,s = x0.shape
	print x0.shape
	for l in range(s):
		temp_list=[]
		#Read the values into a list
		for i in range(p):
			for j in range(q):
				for k in range(r):
					temp_list.append(x0[i][j][k][l])

		temp_list_centroids, templist_index = kmeans_data(temp_list, (p*q*r/4))
		#Replace the values in temp list
		for i,item in enumerate(temp_list):
			temp_list[i] = temp_list_centroids[templist_index[i]]
		#Write Back to the original ndarray
		for i in range(p):
			for j in range(q):
				for k in range(r):
					x0[i][j][k][l] = temp_list[i*q*r+j*r+k]

	return_list.append(x0)

	#Do nothing for biases
	return_list.append(x1)

	return return_list

#Convert 1 fc layer to weight sharing with kmeans
def convert_kmeans_fc(x):
	x0 = x[0] #Filters
	x1 = x[1] #Biases
	return_list = []

	p,q = x0.shape
	temp_list=[]
	#Read the values into a list
	for i in range(p):
		for j in range(q):
			temp_list.append(x0[i][j])

	temp_list_centroids, templist_index = kmeans_data(temp_list, p*q/4 )
	#Replace the values in temp list
	for i,item in enumerate(temp_list):
		temp_list[i] = temp_list_centroids[templist_index[i]]
	#Write Back to the original ndarray
	for i in range(p):
		for j in range(q):
			x0[i][j] = temp_list[i*q+j]

	return_list.append(x0)

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
def flatten_rsc(x):
	p,q,r,s = x.shape
	flat=[]
	for i in range(p):
		for j in range(q):
			for k in range(r):
				flat.append(x[i][j][k][0])
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
	count_bin= np.zeros((500,), dtype=np.int)
	used=[]
	for i in x:
		if(i not in used):
			count = x.count(i)
			count_bin[count]=count_bin[count]+1
			used.append(i)
	return count_bin
