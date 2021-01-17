# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 16:53:03 2020

@author: kshit
"""


import csv
import random
import math
import operator

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	

# prepare data
trainingSet=[]
testSet=[]
accuracy=[9999,9999,9999,9999,9999,9999,9999,9999,9999,9999]
split = 0.85
loadDataset('iris.csv', split, trainingSet, testSet)
print('Train set: ' + repr(len(trainingSet)))
print('Test set: ' + repr(len(testSet)))
# generate predictions
predictions=[]
k = 5

#k1 = 15
#k2 = 25
#k3 = 35
#k4 = 45
#k5 = 1
for x in range(len(testSet)):
	neighbors = getNeighbors(trainingSet, testSet[x], k)
	result = getResponse(neighbors)
	predictions.append(result)
	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy[0] = getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')

    
loadDataset('iris.csv', split, trainingSet, testSet)
print('Train set: ' + repr(len(trainingSet)))
print('Test set: ' + repr(len(testSet)))
# generate predictions
predictions=[]
#k = 7
#k1 = 15
#k2 = 25
#k3 = 35
k5 = 10
#k5 = 1
for x in range(len(testSet)):
	neighbors = getNeighbors(trainingSet, testSet[x], k5)
	result = getResponse(neighbors)
	predictions.append(result)
	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy[1] = getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')

split2 = 0.85
loadDataset('iris.csv', split2, trainingSet, testSet)
print('Train set: ' + repr(len(trainingSet)))
print('Test set: ' + repr(len(testSet)))

# generate predictions
predictions=[]
k1 = 15
#k1 = 15
#k2 = 25
#k3 = 35
#k4 = 45
#k5 = 1
for x in range(len(testSet)):
	neighbors = getNeighbors(trainingSet, testSet[x], k1)
	result = getResponse(neighbors)
	predictions.append(result)
	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy[2] = getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')
split3 = 0.85
loadDataset('iris.csv', split3, trainingSet, testSet)
print('Train set: ' + repr(len(trainingSet)))
print('Test set: ' + repr(len(testSet)))

loadDataset('iris.csv', split, trainingSet, testSet)
print('Train set: ' + repr(len(trainingSet)))
print('Test set: ' + repr(len(testSet)))
# generate predictions
predictions=[]
#k = 7
#k1 = 15
#k2 = 25
#k3 = 35
k6 = 20
#k5 = 1
for x in range(len(testSet)):
	neighbors = getNeighbors(trainingSet, testSet[x], k6)
	result = getResponse(neighbors)
	predictions.append(result)
	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy[3] = getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')

# generate predictions
predictions=[]
#k = 7
#k1 = 15
k2 = 25
#k3 = 35
#k4 = 45
#k5 = 1
for x in range(len(testSet)):
	neighbors = getNeighbors(trainingSet, testSet[x], k2)
	result = getResponse(neighbors)
	predictions.append(result)
	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy[4] = getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')
split4 = 0.85
loadDataset('iris.csv', split4, trainingSet, testSet)
print('Train set: ' + repr(len(trainingSet)))
print('Test set: ' + repr(len(testSet)))

loadDataset('iris.csv', split, trainingSet, testSet)
print('Train set: ' + repr(len(trainingSet)))
print('Test set: ' + repr(len(testSet)))
# generate predictions
predictions=[]
#k = 7
#k1 = 15
#k2 = 25
#k3 = 35
k7 = 30
#k5 = 1
for x in range(len(testSet)):
	neighbors = getNeighbors(trainingSet, testSet[x], k7)
	result = getResponse(neighbors)
	predictions.append(result)
	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy[5] = getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')


# generate predictions
predictions=[]
#k = 7
#k1 = 15
#k2 = 25
k3 = 35
#k4 = 45
#k5 = 1
for x in range(len(testSet)):
	neighbors = getNeighbors(trainingSet, testSet[x], k3)
	result = getResponse(neighbors)
	predictions.append(result)
	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy[6] = getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')
split5 = 0.85
loadDataset('iris.csv', split5, trainingSet, testSet)
print('Train set: ' + repr(len(trainingSet)))
print('Test set: ' + repr(len(testSet)))


loadDataset('iris.csv', split5, trainingSet, testSet)
print('Train set: ' + repr(len(trainingSet)))
print('Test set: ' + repr(len(testSet)))
# generate predictions
predictions=[]
#k = 7
#k1 = 15
#k2 = 25
#k3 = 35
k8 = 40
#k5 = 1
for x in range(len(testSet)):
	neighbors = getNeighbors(trainingSet, testSet[x], k8)
	result = getResponse(neighbors)
	predictions.append(result)
	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy[7] = getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')
    

# generate predictions
predictions=[]
#k = 7
#k1 = 15
#k2 = 25
#k3 = 35
k4 = 45
#k5 = 1
for x in range(len(testSet)):
	neighbors = getNeighbors(trainingSet, testSet[x], k4)
	result = getResponse(neighbors)
	predictions.append(result)
	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy[8] = getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')

    


loadDataset('iris.csv', split5, trainingSet, testSet)
print('Train set: ' + repr(len(trainingSet)))
print('Test set: ' + repr(len(testSet)))
# generate predictions
predictions=[]
#k = 7
#k1 = 15
#k2 = 25
#k3 = 35
k9 = 50
#k5 = 1
for x in range(len(testSet)):
	neighbors = getNeighbors(trainingSet, testSet[x], k9)
	result = getResponse(neighbors)
	predictions.append(result)
	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy[9] = getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')
    
    
text = [5,10,15,20,25,30,35,40,45,50]
import matplotlib.pyplot as plt 
import numpy as np
import sklearn
#%matplotlib inline 

fig, ax = plt.subplots()
width = 0.35

# Add the prior figures to the data for plotting
objects =  list(text)
positive =  list(accuracy)
#res = list(bias_d1.values())

y_pos = np.arange(len(objects))

p1 = ax.bar(y_pos, positive, width, align='center', 
            color=[ 'maroon', 'maroon','maroon','maroon'],alpha=0.5)

#p2 = ax.bar(y_pos+width, res, width, align='center', color=[ 'maroon','maroon','maroon','maroon'],alpha=0.5)

#ax.legend((p1[1], p2[1]), ('Raw', 'Upsampled'))

plt.xticks(y_pos, objects)
'''
plt.ylabel('Minority Counts')
plt.title('Impact of Upsampling ADASYN')'''
 

plt.show()