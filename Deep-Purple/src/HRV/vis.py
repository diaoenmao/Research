
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
  # wearable_feature = pickle.load(open("wearable_API.pkl", "rb" ))
  # RNA_feature = pickle.load(open("RNAdata_API.pkl", "rb" ))
  # feature = []
  # for i in range(len(wearable_feature)):
    # feature_element = np.zeros((4,wearable_feature[0].shape[1]))
    # feature_element[0:3,] = wearable_feature[i]
    # feature_element[-1,] = RNA_feature[i]
    # #print(feature_element[feature_index,:].shape)
    # feature.append(feature_element)   
  #print(feature)
    feature,t,wearable_feature_name = pickle.load(open("feature.pkl", "rb" ))
    hopping_percent = 1/4
    loc = np.int(((24*3+8)*3600)/((1-hopping_percent)*3600))
    print(loc)
    nPeople = len(feature)
    #print(wearable_feature_name)    
  # === understand the data ====
    plotcluster = False
    plotRNA = False
    plotWearable = True
    if(plotcluster):
        s = [1*n**0.5 for n in range(t-1)]
        for i in range(nPeople):
            plt.scatter(feature[i][2,:], feature[i][3,:], marker='o', s=s, facecolors='none', edgecolors='r')
            plt.title(('labeled as %d' % label[i]))
            plt.xlabel('feature 0')
            plt.ylabel('feature 1')
            plt.show()

    if(plotRNA):
        for i in range(nPeople):
            plt.plot(feature[i][3,:])
            plt.title(('RNA labeled as %d' % label[i]))
            plt.xlabel('time')
            plt.ylabel('RNA counts')
            plt.show()
            
    if(plotWearable):
        feature_idx = 1
        for i in range(nPeople):
            plt.plot(feature[i][feature_idx,:])
            plt.title(('%s labeled as %d' % (wearable_feature_name[i][feature_idx],label[i])))
            plt.xlabel('time')
            plt.ylabel(('%s' % wearable_feature_name[i][feature_idx]))
            plt.show()        
        





#%%   
if __name__ == "__main__":
    main()   
