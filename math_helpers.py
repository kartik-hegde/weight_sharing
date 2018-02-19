import numpy as np
from scipy.cluster.vq import vq, kmeans
import collections

#Perform kmeans classification on a given array and return codebook and indices

def kmeans_data(x,centroids):
	kmeans_x = kmeans(x, centroids)
	vq_x	 = vq(x,kmeans_x[0])
	return kmeans_x[0],vq_x[0]


def updateWeightsR(low, high, tempX):
        for i in range(len(tempX)):
                if(tempX[i]>= low and tempX[i] < high):
                        tempX[i] = (low + high)/2.0

def updateWeightsL(low, high, tempX):
        for i in range(len(tempX)):
                if(tempX[i]> low and tempX[i] <= high):
                        tempX[i] = (low + high)/2.0

#Models normal distribution and returns quantized values
def normal_data(x, scale_factor):
        normalX = x
        mu = 0
        var = 0
        numWeights = len(x)
	N = int(numWeights/scale_factor) + 1
        mu = sum(x)/numWeights
        #for i in range(len(uniqueWeights)):
        #       uniqueWeights[i] -= mu
        #       uniqueWeights[i] **=2
        #var = sum(uniqueWeights)/len(uniqueWeights)
        weightCount = {}
        for i in x:
                if i not in weightCount.keys():
                        weightCount.update({i:1})
                else:
                        weightCount[i] += 1
        weightCountSorted = collections.OrderedDict(sorted(weightCount.items()))
        numWeightsInBin = (int)(len(x)/N)
        leftPTR = mu
        rightPTR = mu
        rightIdx = 0
        leftIdx = 0
        keys = weightCountSorted.keys()
        for i in range(len(keys)):
                if(keys[i] >=rightPTR):
                        rightIdx = i
                        break
        for i in range(len(keys)):
                if(keys[len(keys)-i-1] <=leftPTR):
                        leftIdx = len(keys) - i -1
                        break
        if(leftIdx == rightIdx):
                leftIdx -= 1
        for n in range(0, N/2):
                countLeft = 0
                countRight = 0
                while(countRight <= numWeightsInBin and rightIdx < len(keys)-1):
                        countRight += weightCount[keys[rightIdx]]
                        rightIdx += 1
                updateWeightsR(rightPTR, keys[rightIdx], normalX)
                rightPTR = keys[rightIdx]

                while(countLeft <= numWeightsInBin and leftIdx > 0):
                        countLeft += weightCount[keys[leftIdx]]
                        leftIdx -=1
                updateWeightsL(keys[leftIdx], leftPTR, normalX)
                leftPTR = keys[leftIdx]

        return(normalX);

