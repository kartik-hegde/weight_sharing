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

#This function calculates the repeating overlaps in RSC Columns
def calc_repetition_overlap_rsc(x):
	for i,item_column in enumerate(x):
		for j in range(1,len(item_column)):
			repeat_list = overlap_arrays(x[i][0], x[i][j])
			print "In Column" +  str(i) + " with Row" + str(i) + " & Row" + str(j) + " the repetitions are:"
			print "\t\tThere were "+str(len(repeat_list))+" overlapping repetitions"
			print "\t\t\t\t Histogram is: " + str(get_histogram(repeat_list,10))
	
def get_histogram(x, length):
	return_list=[]
	for i in range(1,length):
		return_list.append(x.count(i))
	return return_list

#This function can help find the overlaps in weight sharing, given two arrays
def overlap_arrays(x,y):
	temp_list =[]
	repeat_list=[]
	for i, item in enumerate(x):
		if item not in temp_list:
			indices_x = [j for j,p in enumerate(x) if p == item]
			indices_y = [k for k,q in enumerate(x) if q == item]
			repeat_list.append(len(set(indices_x) & set(indices_y)))
			temp_list.append(item)
	return repeat_list
	
data = np.load("data_kmeans_scaled.npy").item()

#Change this to get the desired data
item = data['conv3'][0]
item_rsc = extract_rsc(item,1)
calc_repetition_overlap_rsc(item_rsc)
