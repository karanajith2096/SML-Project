#This is the python code for Naive Bayes classiifcation
#on MNIST handwritten digits dataset. 

import scipy.io
import numpy as np
import numpy.ma as ma
import math
from scipy.stats import multivariate_normal
#Fucntion to calculate Standard Deviation
def stDev(array):
    mean = np.mean(array)
    var = sum([pow(x - mean, 2) for x in array])/ float(len(array)-1)
    return math.sqrt(var)

#Function to calculate log likelyhood for P(X|Y)
def logLikely(array, mean, std):
    p = 1
    for i in range(len(array)):
        p *= pdf(array[i], mean[i], std[i])
    return p

#Function to calculate Probability Density Function
def pdf(val, mean, std):
    d = val - mean
    if std != 0:
        return math.exp(-1 * (pow(d, 2)/(2 * (pow(std, 2))))) / ((pow(2 * math.pi, 0.5)) * std)
    else:
        return 1
#########################################################################################
#Loading data 
data = scipy.io.loadmat('mnist_data.mat')

#Getting count of the images of each type from training set labels
num_seven = data['trY'][0].tolist().count(0.0)
num_eight = data['trY'][0].tolist().count(1.0)

#Extracting training data images for 7 and 8
train_7 = data['trX'][: num_seven]
train_8 = data['trX'][num_seven :]

#Extracting testing images
test = data['tsX']

#Extracting testing labels
labels = data['tsY']

#Transpose of data
train_7 = train_7.T
train_8 = train_8.T

#Calculating mean and variance for training data
mean_7 = []
for i in range(784):
    mean_7.append(np.mean(train_7[i]))

var_7 = []
for i in range(784):
    var_7.append(np.var(train_7[i]))

std_7 = []
for i in range(784):
    std_7.append(stDev(train_7[i]))

mean_8 = []
for i in range(784):
    mean_8.append(np.mean(train_8[i]))

var_8 = []
for i in range(784):
    var_8.append(np.var(train_8[i]))

std_8 = []
for i in range(784):
    std_8.append(stDev(train_8[i]))


#Calculating priors for each type 
prior_7 = num_seven / (num_seven + num_eight)
prior_8 = 1.0 - prior_7

#Calculating posterior probablity for testing data
out = []
for i in range(2002):
    post_7 = logLikely(test[i], mean_7, std_7) + np.log(prior_7)
    post_8 = logLikely(test[i], mean_8, std_8) + np.log(prior_8)
    if post_7 > post_8:
        out.append(0.0)
    else:
        out.append(1.0)

#Cross checking predicted labels with actual labels
count_7 = 0
count_8 = 0
i = 0
while i < 1028:
    if out[i] == labels[0, i]:
        count_7 += 1
    i += 1

while i < 2002:
    if out[i] == labels[0, i]:
        count_8 += 1
    i += 1

#Calculating accuracy
acc_7 = count_7 / 1028
acc_8 = count_8 / 974
acc = (count_7 + count_8) / 2002

print("Accuracy for '7': " + str(acc_7))
print("Accuracy for '8': " + str(acc_8))
print("Overall accruacy: " + str(acc))
