import numpy as np

#This function returns an RSC in following shape for Kth filter
#[ [ [C column data for 0th row] .. [C column data for Rth row] ] .. S of these ]
def extract_rsc(x,K):
	print "Extracting an RSC of shape "+ str(x.shape)
	return_list = []
	p,q,r,s = x.shape
	#Read the values into a list
	for j in range(q):
		column_list=[]
		for i in range(p):
			row_list=[]
			for k in range(r):
				row_list.append(x[i][j][k][K])
			column_list.append(row_list)
		return_list.append(column_list)
	return return_list

#This function returns an RSC in following shape for Kth filter
#[ [RS data for 0th filter] .. [RS column data for Kth filter] ]
def extract_rsk(x,C):
	print "Extracting an RSC of shape "+ str(x.shape)
	p,q,r,s = x.shape
	#Read the values into a list
	column_list=[]
	for k in range(s):
		row_list=[]
		for j in range(p):
			for i in range(q):	
				row_list.append(x[i][j][C][k])
		column_list.append(row_list)
	return column_list

#This function calculates the repeating overlaps in RSC Columns
def calc_repetition_overlap_rsc(x):
	for i,item_column in enumerate(x):
		for j in range(1,len(item_column)):
			repeat_list = overlap_arrays(x[i][0], x[i][j])
			print "In Column" +  str(i) + " with Row" + str(i) + " & Row" + str(j) + " the repetitions are:"
			print "\t\tThere were "+str(len(repeat_list))+" overlapping repetitions"
			histogram,saved = get_histogram(repeat_list,10)
			print "\t\t\t\t Histogram is: " + str(histogram)
			print "Savings is " + str(saved*1.0/len(x))

#This function calculates the repeating overlaps in RSK
def calc_repetition_overlap_rsk(x):
	for j in range(1,len(x)):
		repeat_list = overlap_arrays(x[0], x[j])
		print "In RSK with Filter" +  str(0) + " and Filter" + str(j) + ", the repetitions are:"
		print "\t\tThere were "+str(len(repeat_list))+" overlapping repetitions"
		histogram,saved = get_histogram(repeat_list,20)
		print "\t\t\t\t Histogram is: " + str(histogram)
		print "Savings is " + str(saved*1.0/len(x))


def get_histogram(x, length):
	return_list=[]
	saved=0
	for i in range(1,length):
		return_list.append(x.count(i))
		
	for i,item in enumerate(return_list):
		saved = saved + item * i
	return return_list,saved

#This function can help find the overlaps in weight sharing, given two arrays
def overlap_arrays(x,y):
	temp_list =[]
	repeat_list_x=[]
	repeat_list_y=[]
	repeat_list =[]
	for i, item in enumerate(x):
		if item not in temp_list and item!=0 :
			indices_x = [j for j,p in enumerate(x) if p == item]
			repeat_list_x.append(indices_x)
			temp_list.append(item)
	temp_list=[]
	for i, item in enumerate(y):
		if item not in temp_list and item!=0:
			indices_y = [j for j,p in enumerate(y) if p == item]
			repeat_list_y.append(indices_y)
			temp_list.append(item)
	for i,item in enumerate(repeat_list_x):
		for j,item_y in enumerate(repeat_list_y):
			repeat_len = len( set(item) & set(item_y) )
			if repeat_len > 1:
				repeat_list.append(repeat_len)
	return repeat_list
	
#This function can help find the overlaps in weight sharing, given three arrays
def overlap_arrays_3(x,y,z):
	temp_list =[]
	repeat_list_x=[]
	repeat_list_y=[]
	repeat_list_z=[]
	repeat_list_xy =[]
	repeat_list_xyz=[]
	for i, item in enumerate(x):
		if item not in temp_list and item!=0 :
			indices_x = [j for j,p in enumerate(x) if p == item]
			repeat_list_x.append(indices_x)
			temp_list.append(item)
	temp_list=[]
	for i, item in enumerate(y):
		if item not in temp_list and item!=0:
			indices_y = [j for j,p in enumerate(y) if p == item]
			repeat_list_y.append(indices_y)
			temp_list.append(item)
	temp_list=[]
	for i, item in enumerate(z):
		if item not in temp_list and item!=0:
			indices_z = [j for j,p in enumerate(z) if p == item]
			repeat_list_z.append(indices_z)
			temp_list.append(item)
	for i,item in enumerate(repeat_list_x):
		for j,item_y in enumerate(repeat_list_y):
			repeat_len = len( set(item) & set(item_y) )
			if repeat_len > 1:
				repeat_list_xy.append(repeat_len)
	for i,item in enumerate(repeat_list_x):
		for j,item_y in enumerate(repeat_list_y):
			for j,item_z in enumerate(repeat_list_z):
				repeat_len = len( set(item) & set(item_y) & set(item_z) )
				if repeat_len > 0:
					repeat_list_xyz.append(repeat_len)

	return repeat_list_xyz

data = np.load("data_8bit_kmeans_32.npy").item()

#Change this to get the desired data
item = data['conv3'][0]

item_rsc = extract_rsc(item,1)
#print get_histogram(overlap_arrays_3(item_rsc[0][0], item_rsc[0][1],item_rsc[0][2]),10)
calc_repetition_overlap_rsc(item_rsc)

#item_rsk = extract_rsk(item,1)
#calc_repetition_overlap_rsk(item_rsk)
