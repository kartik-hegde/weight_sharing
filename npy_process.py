import numpy as np

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

def byte_fixed(x):
	return int(x * 2**8)

def byte_to_32b(x):
	return (x*1.0) / 2**8

def byte_fixed_list(x):
	converted=[]
	for i in x:
		converted.append(int(i * 2**8))
	return converted

def byte_to_32b_list(x):
        converted=[]
	for i in x:
	 converted.append((i*1.0) / 2**8)
	return converted

def flatten_rsc(x):
	p,q,r,s = x.shape
	flat=[]
	for i in range(p):
		for j in range(q):
			for k in range(r):
				flat.append(x[i][j][k][0])
	return flat


def flatten_rsk(x):
	p,q,r,s = x.shape
	flat=[]
	for i in range(p):
		for j in range(q):
			for k in range(r):
				flat.append(x[i][j][0][k])
	return flat

def get_count(x): #receive a flat list
	count_bin= np.zeros((500,), dtype=np.int)
	used=[]
	for i in x:
		if(i not in used):
			count = x.count(i)
			count_bin[count]=count_bin[count]+1
			used.append(i)
	return count_bin
