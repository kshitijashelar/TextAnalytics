# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 12:11:47 2020

@author: kshit
"""

                                                                                                                                                                                                                                                #Taken from https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
# Just added plotting for 3-k cases

import numpy as np
import random
import matplotlib.pyplot as plt

def init_board(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters

def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

def change_coords(array):
    return list(map(list, zip(*array)))

def parse_output(data):
    clusters = data[1]
    points1 = change_coords(clusters[0])
    plt.plot(points1[0], points1[1], 'ro')
    points2 = change_coords(clusters[1])
    plt.plot(points2[0], points2[1], 'g^')
    points3 = change_coords(clusters[2])
    plt.plot(points3[0], points3[1], 'ys')
    centroids = change_coords(data[0])
    plt.plot(centroids[0], centroids[1], 'kx')
    plt.axis([-1.0, 1, -1.0, 1])
    plt.ylabel('Y-Axis')
    plt.xlabel('X-Axis')
    plt.title('Kmeans')
    plt.show()

#data = init_board(15)
'''data = np.array([[ 0.83577267, -0.46120404]
, [ 0.56189562, -0.43979864]
, [ 0.03744518,  0.36043416]
, [-0.25348083, -0.74947791]
, [-0.92163114, -0.76121384]
, [-0.63833866, -0.30691994]
, [-0.83256332, -0.16902132]
, [ 0.33163975,  0.39036818]
, [ 0.70014424,  0.26487845]
, [ 0.61034854, -0.54116575]
, [ 0.29552672,  0.14690514]
, [ 0.55131436, -0.63176047]
, [-0.28767824,  0.95368876]
, [-0.98202436, -0.70664031]
, [-0.45661574, -0.06990991]])'''

data = np.array([[ 0.55 ,  0.65 ]
, [ 0.125,  0.75 ]
, [ 0.65 ,  0.55 ]
, [ 0.65 ,  0.75 ]
, [-0.4  , -0.4 ]
, [-0.45 , -0.45 ]
, [-0.50,  -0.50 ]
, [ 0.01,   0.75 ]
, [ 0.12,   0.85 ]
, [ 0.19    ,   0.95 ]])
print(data)
print(type(data))
out = find_centers(list(data), 3)
parse_output(out)