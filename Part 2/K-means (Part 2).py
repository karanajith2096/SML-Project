#%%
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io
import pandas as pd
import copy

#Picks initial centers according to the second strategy in question
def centers_str2(points, k):
    rand = random.randint(0, len(points) - 1)
    c = points[rand]
    center = []
    center.append(c)
    max_dist = 0
    x = 1
    while x < k:
        d = dict()
        max_dist = 0
        for p in points:
            if p in center:
                continue
            else:
                dist = 0
                for i in center:
                    dist += ( ((p[0] - i[0])**2) + ((p[1] - i[1])**2) )**(1 / 2)
                
                dist = dist / len(center)
                d[dist] = p
                if dist > max_dist:
                    max_dist = dist
        center.append(d[max_dist])
        x += 1
    
    # x = sorted(d, reverse=True)
    # for i in range(k - 1):
    #     center.append(d[x[i]])

    return center
#Clustering of points according to centers
def clustering(data, centers, first_cluster=False):
    for point in data:
        nearest_center = 0
        nearest_center_dist = None
        for i in range(0, len(centers)):
            euclidean_dist = 0
            for d in range(0, 2):
                dist = abs(point[d] - centers[i][d])
                euclidean_dist += dist**2
            euclidean_dist = np.sqrt(euclidean_dist)
            if nearest_center_dist == None:
                nearest_center_dist = euclidean_dist
                nearest_center = i
            elif nearest_center_dist > euclidean_dist:
                nearest_center_dist = euclidean_dist
                nearest_center = i
        if first_cluster:
            point.append(nearest_center)
        else:
            point[-1] = nearest_center
    return data

#Calculation of centroid
def centroid(data, centers):
    centers_updated = []
    for i in range(len(centers)):
        new_center = []
        n_of_points = 0
        sum_pts = []
        for point in data:
            if point[-1] == i:
                n_of_points += 1
                for dim in range(0, 2):
                    if dim < len(sum_pts):
                        sum_pts[dim] += point[dim]
                    else:
                        sum_pts.append(point[dim])
        if len(sum_pts) != 0:
            for dim in range(0, 2):
                new_center.append(sum_pts[dim]/n_of_points)
            centers_updated.append(new_center)
        else:
            centers_updated.append(centers[i])
    return centers_updated

# Trains and returns final list of centriods
def Kmeans(data, k=2, iter=3000):
    #centers = centers_str1(data, k)
    centers = centers_str2(data ,k)

    clustered_data = clustering(data, centers, first_cluster=True)

    temp = []
    for i in range(iter):
        centers = centroid(clustered_data, centers)
        if centers == temp:
            break
        clustered_data = clustering(data, centers, first_cluster=False)
        temp = copy.deepcopy(centers)

    return centers

#Given a point, predicts the cluster that it belongs to
def predict_cluster(point, centers):
    nearest_center = None
    nearest_dist = None
    
    for i in range(len(centers)):
        euclidean_dist = 0
        for d in range(0, 2):
            dist = abs(point[d] - centers[i][d])
            euclidean_dist += dist**2
        euclidean_dist = np.sqrt(euclidean_dist)
        if nearest_dist == None:
            nearest_dist = euclidean_dist
            nearest_center = i
        elif nearest_dist > euclidean_dist:
            nearest_dist = euclidean_dist
            nearest_center = i
        #print('center:',i, 'dist:',euclidean_dist[i])

    return centers[nearest_center]


#%%
#######################################################################################

#Extracting data from given .mat file
data = scipy.io.loadmat('AllSamples.mat')
pts = data['AllSamples']

#Running for all values from k = 2 to k = 10
objective_function = []
for k in range(2, 11):
    #Calling Kmeans for clustering
    centers = Kmeans(pts.tolist(), k)

    #Displaying graphs of points with centroids marked in red
    plt.scatter(pts[ :, 0], pts[ :, 1])
    for i in range(k):
        plt.plot(centers[i][0], centers[i][1], "ro")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Clustering (k = " + str(k) + ")")
    plt.show()

    #Calculating Objective function for each k
    dist = []
    for i in range(k):
        dist.append(0)

    for point in pts:
        center = predict_cluster(point, centers)
        j = centers.index(center)
        dist[j] += ( ((point[0] - center[0])**2) + ((point[1] - center[1])**2) )**(0.5)

    sum = 0
    for i in range(k):
        sum += (dist[i]**2) / 100

    objective_function.append(sum)


#Plotting Objective function for each k
plt.plot(range(2, 11), objective_function, c="white")
plt.ylabel("Objective Function")
plt.xlabel("Number of clusters")
plt.title("Elbow Method(Strategy 2)")
plt.show()


# %%
