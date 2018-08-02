# -*- coding: utf-8 -*-
"""
Created on Mon May 22 2017

@author: jieding  
"""
import numpy as np 
from numpy import concatenate, zeros, mean, var, dot, maximum, exp, array, mat, ravel, diag
from numpy.linalg import inv
import scipy 
import copy
import pickle
from scipy.stats import multivariate_normal 
import matplotlib.pyplot as plt

# === Static param ===
# d: dimension of cluter/test points, e.g. 2 for (R,P)
#d = 2


# t: total time steps 
#t = 4
t = 21 

# loc: change point location for training people
#loc = 1
loc = 11 

# range of time stps to test, testStart,...,testStart+testSpan-1
#testRange = {'testStart':0, 'testSpan':3} 
testRange = {'testStart':11, 'testSpan':4} 

# === introduction of input parameters ===
# label: a list of true 0-1 (integer) sickness indicators of length nPeople  
# feature: a list of length nPeople, each an array of size d (dimension) x t (time steps)
#   dimension 1: RNA data of size 1 x t
#   dimension 2: (combined) wearable data of size 1 x t 
#   ...

# === introduction of output ===
# decision: 0/1 indicating sick/not sick 
# reliability: value on [0,1] indicating the reliability of 0-1 test decision, i.e. posterior probability of choosing the correct 


feature_index = [0,1,2,3]
d = len(feature_index)


# ========== sample use of the output API using synthetic data =========== 
# ========================================================================
def main():
  # test static params
  # d = 2 
  # t = 4
  # loc = 2
  # testRange = {'testStart':2, 'testSpan':1} 

  # API data 
  label = pickle.load(open("labels.pkl", "rb" )).tolist()
  wearable_feature = pickle.load(open("wearable_API.pkl", "rb" ))
  RNA_feature = pickle.load(open("RNAdata_API.pkl", "rb" ))
  feature = []
  for i in range(len(wearable_feature)):
    feature_element = np.zeros((4,wearable_feature[0].shape[1]))
    feature_element[0:3,] = wearable_feature[i]
    feature_element[-1,] = RNA_feature[i]
    #print(feature_element[feature_index,:].shape)
    feature.append(feature_element[feature_index,:])   
  #print(feature)
  nPeople = len(feature)

  # === understand the data ====
  # print("Curve plot of feature 0 and 1:")
  # s = [20*n**2 for n in range(21)]
  # plt.plot(feature[0][0,:], feature[0][1,:], s=s)
  # plt.xlabel('feature 0')
  # plt.ylabel('feature 1')
  # plt.show()


  # Synthetic data
  # label = [0, 0, 1, 1]
  # feature = [mat('0 0.1 0.3 -0.4; 0 0.2 0.5 -0.1'), mat('0 -0.1 -0.3 0.4; 0 -0.7 0.2 0.1'), mat('1 1.1 1.3 -1.4; 0.6 0.7 1.5 1.1'), mat('1 1.1 1.3 -1.4; 0.6 0.7 1.5 1.1')]
  #print(feature[0].shape)
  
  # ============== run for a fixed threshold ==============
  thresh = 1
  loocv = getLOOCV(feature, label, loc, testRange, thresh) 
  
  print("=== Result for a fixed thresh ===:")
  print("Our decisions:")
  print(loocv['decisions'])
  print("True labels:")
  print(label) 
  print("Our success rate:")
  print(loocv['success'])
  print("Our reliabilities:")
  print(loocv['reliabilities'])


  # ============== run for different thresholds and plot ROC ==============
  # rates of misDectect, falseAlarm, successful decision, and average reliability  of varying thresh 
  misDectect = [0]
  falseAlarm = [1]
  success = []
  reliability = []

  # plot ROC curve
  r = np.linspace(0, 10, num=200) 
  for thresh in np.nditer(r): 
    loocv = getLOOCV(feature, label, loc, testRange, thresh) 
    misDectect.append( loocv['misDectect'] )
    falseAlarm.append( loocv['falseAlarm'] ) 
    success.append( loocv['success'] ) 
  misDectect.append(1)
  falseAlarm.append(0)

  print("=== Result for varying thresh ===:")
  print("Our best success rate:")
  print(np.max(success))

  print("Our ROC curve (see plot):")
  plt.plot(falseAlarm, misDectect, 'r--', linewidth = 2.0)
  plt.axis([0, 1, 0, 1])
  plt.xlabel('probability of false alarm')
  plt.ylabel('probability of mis-detection')
  plt.show()

# ============================== subroutines ========================== 
# ========================================================================


# the function obtain the mean and covariance of d-dimensional points before and after change points 
# featureTrain: a list of training people, sublist of feature
def getTrain(featureTrain, labelTrain, loc):
  # num of test people 
  nPeopleTrain = len(featureTrain)

  dat0 = zeros((d, 0)) 
  dat1 = zeros((d, 0)) 
  for people in range(nPeopleTrain):
    f = featureTrain[people]

    # first extract data of size d x (nPeopleTrain x loc) before change: 0,1,...,loc-1 (H0)
    # should be size zeros((d, nPeopleTrain x loc))
    if labelTrain[people] > 0: #if sick
      dat0 = concatenate((dat0, f[:, 0:loc-1]), axis=1) 
      dat1 = concatenate((dat1, f[:, loc:t]), axis=1) 
    else: 
      # extract data from all time, who has no sickness
      #print(dat0.shape)
      #print(f.shape)
      dat0 = concatenate((dat0, f), axis=1) 

  # compute prior of getting sick  
    prior1 = 1.0 * sum(labelTrain) / len(labelTrain)  

  # compute mean 
  m0 = mean(dat0, axis=1)
  v0 = np.cov(dat0)
  #v0 = diag(ravel(var(dat0, axis=1))) #assume independence 
  m1 = mean(dat1, axis=1)
  v1 = np.cov(dat1)
  #v1 = diag(ravel(var(dat1, axis=1))) #assume independence 

  return {'m0': m0, 'v0': v0, 'm1': m1, 'v1': v1, 'prior1': prior1}

# the function make binary decision (along with reliability measure) for one people, using trained information 
# datTest:  an array of size d (dimension) x t (time steps)
# time: decision time step, time = 0,1...,t-1
# trainRes: result from getTrain 
# thresh: threshold for (sequential) decision making, default should be one, change it for ROC plotting 
# return decision as boolean, and reliability as a value on [0,1]
def getDecision(datTest, trainRes, testRange, thresh):
  lik0 = 0
  lik1 = 0
  for time in range(testRange['testStart'], testRange['testStart']+testRange['testSpan']):
    a = 0
    #print(multivariate_normal.logpdf(ravel(datTest[:, time]), mean=ravel(trainRes['m0']), cov=trainRes['v0']))
    lik0 =  lik0 + multivariate_normal.logpdf(ravel(datTest[:, time]), mean=ravel(trainRes['m0']), cov=trainRes['v0']) 
    lik1 =  lik1 + multivariate_normal.logpdf(ravel(datTest[:, time]), mean=ravel(trainRes['m1']), cov=trainRes['v1'])
  lik0 = lik0 - maximum(lik0, lik1)
  lik1 = lik1 - maximum(lik0, lik1)
  ratio10 = exp(lik1-lik0) * trainRes['prior1'] / (1-trainRes['prior1'])
  decision = (ratio10 > thresh) # true if H1 is preferred 
  reliability = maximum(ratio10,1) / (maximum(ratio10,1) + 1) #[0.5,1]
  #reliability = 2 * (reliability - 0.5) # rescale to [0,1]

  return {'decision': decision, 'reliability': reliability}

def getLOOCV(feature, label, loc, testRange, thresh): 
  # number of training+testing people 
  nPeople = len(feature)

  # (integer) value indicating num of decisions 
  success = 0
  misDectect = 0
  falseAlarm = 0

  # (double) vector of decision decisions and reliabilities  
  decisions = zeros(nPeople)
  reliabilities = zeros(nPeople)

  for people in range(nPeople):
    # assign training feature and test feature simultaneously 
    featureTrain = copy.deepcopy(feature) 
    datTest = featureTrain.pop(people) 

    # assign true label 
    labelTrain = copy.deepcopy(label)
    
    del labelTrain[people]

    # train 
    trainRes = getTrain(featureTrain, labelTrain, loc) 

    # test 
    testRes = getDecision(datTest, trainRes, testRange, thresh)

    # store results 
    if (bool(label[people]) and not testRes['decision']):
      misDectect += 1
    elif (not bool(label[people]) and testRes['decision']):
      falseAlarm += 1
    else:
      success += 1 

  # compute rates   
  misDectect = misDectect * 1.0 / sum(label) 
  falseAlarm = falseAlarm * 1.0 / (nPeople - sum(label))
  success = success * 1.0 / nPeople 
  decisions[people] = testRes['decision'] 
  reliabilities[people] = testRes['reliability'] 

  return {'success': success, 'misDectect': misDectect, 'falseAlarm': falseAlarm, 'decisions': decisions, 'reliabilities': reliabilities}


#%%   
if __name__ == "__main__":
    main()   
