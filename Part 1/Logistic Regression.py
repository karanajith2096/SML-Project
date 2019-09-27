#This is the Python code for Logistic Regression 
#on MNIST handwritten digits dataset.


import scipy.io
import numpy as np
import numpy.ma as ma
from sklearn.naive_bayes import GaussianNB

#Sigmoid function
def sigmoid(score):
    return 1 / (1 + np.exp(-score))

#Logistic regression function to do the training of model
def LogReg(features, t, num_steps, rate):
    w = np.zeros(features.shape[1])
    for step in range(num_steps):
        score = np.dot(features, w)
        p = sigmoid(score)

        #Using gradient to update weights
        err_out = t - p
        grad = np.dot(features.T, err_out)
        w += rate * grad
        
    return w


#############################################################
#Loading data
data = scipy.io.loadmat('mnist_data.mat')

#Extracting training data
train_data = data['trX']
train_labels = data['trY'][0]

#Extracting testing data
test_data = data['tsX']
test_labels = data['tsY'][0]

#Calculating Weight
weight = LogReg(train_data, train_labels, 7000, 2e-3)

#Calcuating final probability and the predicted labels
final = np.dot(test_data, weight)
p = np.round(sigmoid(final))

#Cross checking predicted labels with actual labels
count_7 = 0
count_8 = 0
i = 0
while i < 1028:
    if p[i] == test_labels[i]:
        count_7 += 1
    i += 1

while i < 2002:
    if p[i] == test_labels[i]:
        count_8 += 1
    i += 1

#Calculating accuracy
acc_7 = count_7 / 1028
acc_8 = count_8 / 974
acc = (count_7 + count_8) / 2002

print("Accuracy for '7': " + str(acc_7))
print("Accuracy for '8': " + str(acc_8))
print("Overall Accuracy: " + str(acc))






