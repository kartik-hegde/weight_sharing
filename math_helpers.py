import numpy as np
from scipy.cluster.vq import vq, kmeans


#Perform kmeans classification on a given array and return codebook and indices

def kmeans_data(x,centroids):
	kmeans_x = kmeans(x, centroids)
	vq_x	 = vq(x,kmeans_x[0])
	return kmeans_x[0],vq_x[0]

