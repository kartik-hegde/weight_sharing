import numpy as np
import csv
import random
from npy_process import *
from overlapped_sharing import *
import operator
import argparse
import cPickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument(
	#'--weights_File', type=str, default='/home/rohita2/Resnet-Retrain/weights_cifar10_orig.npy',
	'--weights_File', type=str, default='/home/rohita2/caffe/AlexNet_INQ.npy',
        help='The npy file containing the weights of the network')

parser.add_argument(
	'--has_Bias', action='store_true', default=False,
        help='Flag to indicate if the dumped weights contain bais in each layer')

parser.add_argument(
	'--do_KMeans', action='store_true', default=False,
        help='do you want to perform K-Means on the data?')

parser.add_argument(
	'--iteration', type=int, default=0,
        help='The iteration number in case of multiple retrain cycle. To help dumping analyzing weights for individual retrain cycle')

parser.add_argument(
	'--to_8bits', action='store_true', default=False,
        help='do you want to convert the data to 8-bits?')

parser.add_argument(
	'--to_16bits', action='store_true', default=False,
        help='do you want to convert the data to 16-bits?')

FLAGS = parser.parse_args()

def get_Indices(x):
	used = []
	index = []
	for i in x:
		if i not in used:
			indices = np.where(x == i)[0]
			index.append(indices)
	return(index)



def KCRS2RSCK(x):
	K, C, R, S = np.shape(x)
	x1 = np.zeros([R,S,C,K])
	for k in range(K):
		for c in range(C):
			for r in range(R):
				for s in range(S):
					x1[r][s][c][k] = x[k][c][r][s]
	
	return x1


def get_Repetitions_RSK(x,y):
	index_x = get_Indices(x)	
	index_y = get_Indices(y)
	overlap = []
	for i in range(len(index_x)):
		mx = 1
		for j in range(len(index_y)):
			if(len(set(index_x[i]) & set(index_y[j])) > mx):
				mx = len(set(index_x[i]) & set(index_y[j]))
		overlap.append(mx)
	print(overlap)
	return(overlap)
#def get_Unique_Num(x):
	
def get_Repetitions_RSC(x):
	rep = {}
	zero_Count = 0
	for i in x:
		if i!= 0.0:
			if i not in rep.keys():
				rep.update({i:1})
			else:
				rep[i] +=1
		else:
			zero_Count +=1
	#value = rep.values()
	#count=[]
	#used = []
	#for i in value:
	#	if i not in used:
	#		count.append((value.count(i),i))
	#		used.append(i)
	#saving = 1-((len(rep)*1.0)/sum(rep.values()))
	#sorted_Count = sorted(count, key=operator.itemgetter(1))
	#return(sorted_rep, saving)
	idxx=0
	sorted_Count = get_count(x)
	for idx, item in enumerate(sorted_Count[::-1]):
		if item!=0:
			idxx = idx
			break
	last = len(sorted_Count)-idxx
	sorted_Count = sorted_Count[0:last]
	sorted_Count[0] = zero_Count
	return(sorted_Count)

def flatten_Slice(RSC, x):
	p,q,r = np.shape(RSC)
	flat = []
	for i in range(q):
		for j in range(r):
			flat.append(RSC[x][i][j])
	return(flat)

#def overlap_slices(RSC, r1, r2):
#	p,q,r = np.shape(RSC)
#	for i in range

def get_RSCt(x, K, Ct):
	return_list = []
	p,q,r,s = x.shape
	for i in range(p):
		row_list = []
		for j in range(q):
			column_list = []
			for k in range(Ct):
				column_list.append(x[i][j][k][K])
			row_list.append(column_list)
		return_list.append(row_list)
	return return_list

def get_Repetitions_CCR(x):
	size = max(x)+1 if len(x)>0 else 1
	sorted_Count = np.zeros(size, dtype=int)
	for i in x:
		sorted_Count[i] +=1
	idxx=0
	#sorted_Count = get_count(x)
	for idx, item in enumerate(sorted_Count[::-1]):
		if item!=0:
			idxx = idx
			break
	last = len(sorted_Count)-idxx
	sorted_Count = sorted_Count[0:last]
	sorted_Count[0] = 0
	return(sorted_Count)


	#print(np.shape(x))
	#p,q,r = np.shape(x)
	#row = []
	#for i in range(p-1):
	#	for j in range(q):
	#		print(i,j)
	#		print(overlap_arrays(x[i][j],x[i+1][j]))
	

def analyze_Layers_CCR(data):
	file_Name = 'analysis_cifar_CCR_K'+str(int(FLAGS.do_KMeans))+'_8b'+str(int(FLAGS.to_8bits))+'_16b'+str(int(FLAGS.to_16bits))+'.csv'
	with open(file_Name, 'wb') as csvfile:
        	writer = csv.writer(csvfile, delimiter=',')
		for keys in data.keys():
			if 'conv' in keys:
				print('analyzing ',keys,'of size',np.shape(data[keys]))
				a,p,q,r,s = np.shape(data[keys])
				for k in range(s):
					filter_size = str(p)+'x'+str(q)+'x'+str(r)
					RSC = get_RSCt(data[keys][0],k,r)
					for i in range(p-1):
						for j in range(q):
							row = [keys, 'Filter#'+str(k)+'_'+filter_size]
							row.append('('+str(i)+','+str(j)+')('+str(i+1)+','+str(j)+')')
							overlap = overlap_arrays(RSC[i][j],RSC[i+1][j])
							overlap = get_Repetitions_CCR(overlap)
							for item in overlap:
								row.append(item)
							writer.writerow(row)


def flatten_rsct(x):
	p,q,r = np.shape(x)
	flat=[]
	for i in range(p):
		for j in range(q):
			for k in range(r):
				flat.append(x[i][j][k])
	return flat




def analyze_Layers_FCR(data,CT):
	#file_Name = 'analysis_cifar_FCR_K'+str(int(FLAGS.do_KMeans))+'_8b'+str(int(FLAGS.to_8bits))+'_16b'+str(int(FLAGS.to_16bits))+'.csv'
	file_Name = 'analysis_FCR_resnet_INQ.csv'
	with open(file_Name, 'wb') as csvfile:
        	writer = csv.writer(csvfile, delimiter=',')
		for keys in data.keys():
			if 'conv' in keys:
				print('analyzing ',keys,'of size',np.shape(data[keys][0]))
				#a,p,q,r,s = np.shape(data[keys])
				data[keys][0] = KCRS2RSCK(data[keys][0])
				p,q,r,s = np.shape(data[keys][0])
				filter_size = str(p)+'x'+str(q)+'x'+str(r)
				for k in range(s/2):
					#size = group_Size
					#K=[]
					#if(len(filters)<group_Size):
					#	size = len(filters)
					#	K = random.sample(filters, size)
					#else:
					#	K = random.sample(filters, size)
						Ct=CT
					#R=r
					#while R>0:
						if Ct > r:
							Ct=r
						row = [keys, 'Filter#'+str(k)+'_'+filter_size]
						row.append('('+str(2*k)+','+str(2*k+1)+')')
						RSCt1 = get_RSCt(data[keys][0], 2*k, Ct)
						RSCt2 = get_RSCt(data[keys][0], 2*k+1, Ct)
						flat1 = flatten_rsct(RSCt1)
						flat2 = flatten_rsct(RSCt1)
						overlap = overlap_arrays(flat1, flat2)
						percent=(sum(overlap)*100.0)/(r*s*Ct)
						row.append(str(percent)+'% ')
						overlap = get_Repetitions_CCR(overlap)
						for item in overlap:
							row.append(item)
						writer.writerow(row)

						#for k1 in range(len(K)-1):
						#	flat1 = flatten_rsc(data[keys][0], K[k1])
						#	for k2 in range(k1+1, len(K)):
						#		row = [keys, 'Filter'+'_'+filter_size]
						#		row.append('('+str(K[k1])+','+str(K[k2])+')')
						#		flat2 = flatten_rsc(data[keys][0], K[k2])
						#		overlap = overlap_arrays(flat1, flat2)
						#		overlap = get_Repetitions_CCR(overlap)
						#		for item in overlap:
						#			row.append(item)
						#		writer.writerow(row)
						#for i in K:
						#	filters.remove(i)
	

def analyze_Layers_Sliced_CCR(data):
	#file_Name = 'analysis_cifar_Sliced_CCR_K'+str(int(FLAGS.do_KMeans))+'_8b'+str(int(FLAGS.to_8bits))+'_16b'+str(int(FLAGS.to_16bits))+'.csv'
	file_Name = 'test_CCR_INQ.csv'
	with open(file_Name, 'wb') as csvfile:
        	writer = csv.writer(csvfile, delimiter=',')
		for keys in data.keys():
			if 'conv' in keys:
				data[keys][0] = KCRS2RSCK(data[keys][0])
				p,q,r,s = np.shape(data[keys][0])
				print('analyzing ',keys,'of size',np.shape(data[keys]))
				#a,p,q,r,s = np.shape(data[keys])
				for k in range(1):
					filter_size = str(p)+'x'+str(q)+'x'+str(r)
					RSC = get_RSCt(data[keys][0],k,r)
					#for i in range(p):
					#	row = [keys, 'Filter#'+str(k)+'_'+filter_size, 'row#'+str(i)]
					#	flat1 = flatten_Slice(RSC, i)
					#	count = get_Repetitions_RSC(flat1)
					#	for item in count:
					#		row.append(item)
					#	writer.writerow(row)
					for i in range(p-1):
						for j in range(i+1,p):
							row = [keys, 'Filter#'+str(k)+'_'+filter_size]
							row.append('('+str(i)+','+str(j)+')')
							flat1 = flatten_Slice(RSC, i)
							print(flat1.count(0))
							flat2 = flatten_Slice(RSC, j)
							overlap = overlap_arrays(flat1, flat2)
							percent=(sum(overlap)*100.0)/(q*r)
							row.append(str(percent)+'% ')
							overlap = get_Repetitions_CCR(overlap)
							for item in overlap:
								row.append(item)
							writer.writerow(row)


def analyze_Layers_ResNet_RSC(data):
	#file_Name = 'analysis_cifar_RSC_K'+str(int(FLAGS.do_KMeans))+'_8b'+str(int(FLAGS.to_8bits))+'_16b'+str(int(FLAGS.to_16bits))+'.csv'
	file_Name = 'frequency_analysis_resnet_INQ.csv'
	resnet_layers=dict()
	#resnet_layers['128']=[]
	#resnet_layers['256']=[]
	#resnet_layers['64']=[]
	#resnet_layers['512']=[]
	resnet_layers['128']=dict()
	resnet_layers['256']=dict()
	resnet_layers['64']=dict()
	resnet_layers['512']=dict()
	big_dict={}
	with open(file_Name, 'wb') as csvfile:
        	writer = csv.writer(csvfile, delimiter=',')
		#flag = 0
		for keys in data.keys():
			if 'conv' in keys and 'layer' in keys and 'expand' not in keys:
			#if 'layer_128_2_conv1' in keys:
				print(keys, np.shape(data[keys][0]))
				data[keys][0] = KCRS2RSCK(data[keys][0])
				p,q,r,s = np.shape(data[keys][0])
				#s,r,p,q = np.shape(data[keys][0])
				split_key = keys.split("_")
				avg = 0
				avg_zero=0
				for k in range(s):
					filter_size = str(p)+'x'+str(q)+'x'+str(r)
					#print(np.shape(data[keys][0]))
					flat = flatten_rsc(data[keys][0], k)
					#layer = str(split_key[1])+"_"+str(split_key[2])+"_"+str(split_key[3])+"_"+str(s)
					#layer = str(split_key[1])+"_"+str(split_key[3])+"_"+str(s)
					layer = str(split_key[1])+"_"+str(split_key[3])
					#FOr per-weight analysis
					if layer not in big_dict.keys():
						big_dict.update({layer:[flat]})
					else:
						big_dict[layer] += [flat]
					#for non per-weight analysis 
					#rep ={}
					#for i in set(flat):
					#	rep[i]=flat.count(i)
					#temp = [rep]
					#if layer not in big_dict.keys():
					#	big_dict.update({layer:temp})
					#else:
					#	big_dict[layer].append(rep) 
					#print(len(flat))
					num_Unique = len(set(flat))
					num_non_zero = num_Unique-1 if 0 in flat else num_Unique
					if num_non_zero==0:
						num_non_zero+=1
					#count = get_Repetitions_RSC(flat)
					zero_count = flat.count(0)
					count=[]
					for i in set(flat):
						count.append(flat.count(i))
					avg += ((len(flat)-zero_count)*1.0)/(num_non_zero)
					avg_zero += zero_count
					row = [keys, 'Filter#'+str(k)+'_'+filter_size, 'Num_Unique#'+str(num_Unique)]
					for i in count:
						row.append(i)
					writer.writerow(row)
						#writer.writerow([keys, 'Filter#'+str(k)+'_'+filter_size, count])
				avg /= s
				avg_zero /= s
				#resnet_layers[split_key[1]].append((split_key[2],split_key[3],avg,avg_zero))
			#	if split_key[3] not in resnet_layers[split_key[1]].keys():
			#		resnet_layers[split_key[1]].update({split_key[3]:(avg,avg_zero)})
			#	else:
			#		resnet_layers[split_key[1]][split_key[3]] = tuple(map(operator.add, resnet_layers[split_key[1]][split_key[3]],(avg,avg_zero)))
				if split_key[3] not in resnet_layers[split_key[1]].keys():
					resnet_layers[split_key[1]].update({split_key[3]:[[avg],[avg_zero]]})
				else:
					resnet_layers[split_key[1]][split_key[3]][0].append(avg)
					resnet_layers[split_key[1]][split_key[3]][1].append(avg_zero)
	for keys in resnet_layers.keys():
		for k in resnet_layers[keys].keys():
			#if keys=='64':
			resnet_layers[keys][k]= [(np.mean(resnet_layers[keys][k][0]),np.std(resnet_layers[keys][k][0])),(np.mean(resnet_layers[keys][k][1]),np.std(resnet_layers[keys][k][1]))]
			#if keys=='128':	
			#	resnet_layers[keys][k]=(resnet_layers[keys][k][0]/4.0,resnet_layers[keys][k][1]/4.0) 	
			#if keys=='256':	
			#	resnet_layers[keys][k]=(resnet_layers[keys][k][0]/6.0,resnet_layers[keys][k][1]/6.0) 	
			#if keys=='512':	
			#	resnet_layers[keys][k]=(resnet_layers[keys][k][0]/3.0,resnet_layers[keys][k][1]/3.0) 	
	#for keys in resnet_layers.keys():
	#	for k in resnet_layers[keys].keys():
	#		if keys=='64':
	#			resnet_layers[keys][k]=(resnet_layers[keys][k][0]/3.0,resnet_layers[keys][k][1]/3.0) 	
	#		if keys=='128':	
	#			resnet_layers[keys][k]=(resnet_layers[keys][k][0]/4.0,resnet_layers[keys][k][1]/4.0) 	
	#		if keys=='256':	
	#			resnet_layers[keys][k]=(resnet_layers[keys][k][0]/6.0,resnet_layers[keys][k][1]/6.0) 	
	#		if keys=='512':	
	#			resnet_layers[keys][k]=(resnet_layers[keys][k][0]/3.0,resnet_layers[keys][k][1]/3.0) 	
	print("\n\nprinting resnet_layers")
	print(resnet_layers)
	print("\n\ndone printing resnet_layers")
	count_dict={}
	#for per-weight analysis
	#for keys in big_dict.keys():
	#	temp_dict={}
	#	#list dict for each layer
	#	layer_count=big_dict[keys]
	#	#print(layer_count)
	#	#for each filter there is a dict 
	#	for i in layer_count:
	#		#for each unique weight
	#		for k in i.keys():
	#			if k not in temp_dict.keys():
	#				temp_dict.update({k:[i[k]]})
	#			else:
	#				temp_dict[k].append(i[k])
	#	stddev=[]
	#	for keyss in temp_dict.keys():
	#		stddev.append((np.std(temp_dict[keyss]),np.mean(temp_dict[keyss])))
	#	count_dict[keys]=stddev
	#For non-per-weight analysis
	#for keys in big_dict.keys():
	#	size = len(big_dict[keys])
	#	rep_zero=[]
	#	for i in range(size):
	#		rep_zero.append(big_dict[keys][i].count(0))
	#	flat = big_dict[keys]
	#	k = int(keys.split("_")[3])
	#	count=[]
	#	for i in set(flat):
	#		if i is not 0:
	#			count.append(flat.count(i))
	#	count = [(1.0*i/k) for i in count]
	#	count_dict[keys] = np.std(count)
	#	print(count)
	#	print(keys, k,np.std(count), np.mean(count))
	##print(count_dict)
	#with open("frequency_analysis.csv",'wb') as csvfile:
        #	writer = csv.writer(csvfile, delimiter=',')
	#	for keys in count_dict.keys():
	#		print('\n')
	#		print(keys, count_dict[keys])
	#		temp = []
	#		for i in count_dict[keys]:
	#			temp = [keys,i[0],i[1]]
	#			writer.writerow(temp)
				



def analyze_Layers_RSC(data):
	#file_Name = 'analysis_cifar_RSC_K'+str(int(FLAGS.do_KMeans))+'_8b'+str(int(FLAGS.to_8bits))+'_16b'+str(int(FLAGS.to_16bits))+'.csv'
	file_Name = 'analysis_resnet_INQ.csv'
	with open(file_Name, 'wb') as csvfile:
        	writer = csv.writer(csvfile, delimiter=',')
		#flag = 0
		for keys in data.keys():
			if 'conv' in keys and 'layer' in keys and 'expand' not in keys:
				print(keys, np.shape(data[keys][0]))
				data[keys][0] = KCRS2RSCK(data[keys][0])
				p,q,r,s = np.shape(data[keys][0])
				#s,r,p,q = np.shape(data[keys][0])
				for k in range(s):
					filter_size = str(p)+'x'+str(q)+'x'+str(r)
					print(np.shape(data[keys][0]))
					flat = flatten_rsc(data[keys][0], k)
					print(len(flat))
					num_Unique = len(set(flat))
					#if flag==0:
					count = get_Repetitions_RSC(flat)
					row = [keys, 'Filter#'+str(k)+'_'+filter_size, 'Num_Unique#'+str(num_Unique)]
					for i in count:
						row.append(i)
					writer.writerow(row)
						#writer.writerow([keys, 'Filter#'+str(k)+'_'+filter_size, count])
					#flag=1


def analyze_alexnet_RSC(data):
  file_name = 'analysis_alexnet_INQ.csv'
  with open(file_name, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for keys in data.keys():
      if 'conv' in keys:
        print(keys, np.shape(data[keys][0]))
        data[keys][0] = KCRS2RSCK(data[keys][0])
        p,q,r,s = np.shape(data[keys][0])
        zero_count = []
        avg_repetition = []
        for k in range(s):
          filter_size = str(p)+'x'+str(q)+'x'+str(r)
          print(np.shape(data[keys][0]))
          flat = flatten_rsc(data[keys][0], k)
          num_Unique = len(set(flat))
          num_Unique = num_Unique -1 if 0 in flat else num_Unique
          if num_Unique == 0:
            num_Unique = 1
          avg_repetition.append(len(flat)/(1.0*num_Unique))
          zero_count.append(flat.count(0))
          #count = get_Repetitions_RSC(flat)
          #for i in count:
          #	row.append(i)
          	#writer.writerow([keys, 'Filter#'+str(k)+'_'+filter_size, count])
        row = [keys, 'Filter#'+str(k)+'_'+filter_size, 'Num_Unique#'+str(num_Unique)]
        row.append(np.mean(avg_repetition))
        row.append(np.mean(zero_count))
        writer.writerow(row)



data = np.load(FLAGS.weights_File).item()
if FLAGS.to_8bits:
	print("\nQuantizing to 8-bits\n")
	data = convert_npy_8b(data, FLAGS.has_Bias)
if FLAGS.to_16bits:
	print("\nQuantizing to 16-bits\n")
	data = convert_npy_16b(data, FLAGS.has_Bias)
if FLAGS.do_KMeans:
	print("\nPerforming K-Means\n")
	data = convert_npy_kmeans(data, 4, False, False)
	#data = convert_npy_normal(data, 16, False)
#np.save("weights_cifar_quantized.npy",data)
#np.save("weights_cifar_normalized.npy",data)
#analyze_Layers_ResNet_RSC(data)
#analyze_Layers_RSC(data)
#analyze_Layers_FCR(data,128)
#analyze_Layers_Sliced_CCR(data)
analyze_alexnet_RSC(data)
