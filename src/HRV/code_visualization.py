
import numpy as np 
from numpy import concatenate, zeros, mean, var, dot, maximum, exp, array, mat, ravel, diag
from numpy.linalg import inv
import scipy 
import copy
import pickle
import matplotlib.pyplot as plt



# ========================================================================
def main():

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
    feature.append(feature_element)   
  #print(feature)
  nPeople = len(feature)

  # === understand the data ====
  print("Curve plot of feature 0 and 1:")
  s = [2*n**1.5 for n in range(21)]
  for i in range(nPeople):
    plt.scatter(feature[i][1,:], feature[i][3,:], s=s)
    plt.xlabel('feature 0')
    plt.ylabel('feature 1')
    plt.show()




#%%   
if __name__ == "__main__":
    main()   
