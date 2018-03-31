import numpy as np
from scipy import stats
from sl import *
import pandas

def ttest(x1,x2):
    l1 = load(x1)
    l2 = load(x2)
    n = len(l1)
    t = np.zeros(n)
    p = np.zeros(n)
    for i in range(n):
        #print(l2[i].values.flatten())
        cur_l1 = l1[i].flatten()
        cur_l2 = l2[i].values.flatten()
        t[i],p[i] = stats.ttest_rel(cur_l1,cur_l2)
    return t,p
    
def getlabel():
    labeldir = './label.pkl'
    label = np.array(load(labeldir))
    return label
    
def main():
    EDA_pred = './pred_EDA.pkl'
    EDA_true = './true_EDA.pkl'
    HR_pred = './pred_HR.pkl'
    HR_true = './true_HR.pkl'
    TEMP_pred = './pred_TEMP.pkl'
    TEMP_true = './true_TEMP.pkl'
    label = getlabel()
    print(label)
    EDA_t,EDA_p = ttest(EDA_pred,EDA_true)
    HR_t,HR_p = ttest(HR_pred,HR_true)
    TEMP_t,TEMP_p = ttest(TEMP_pred,TEMP_true)
    print(EDA_p)
    print(HR_p)
    print(TEMP_p)
    print(EDA_p>0.05)
    print(HR_p>0.05)
    print(TEMP_p>0.05)

    
if __name__ == '__main__':
    main()