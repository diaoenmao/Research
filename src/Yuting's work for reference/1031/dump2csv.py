import numpy as np
import os
from sl import *


def main():
    EDA_pred = './pred_EDA.pkl'
    EDA_true = './true_EDA.pkl'
    HR_pred = './pred_HR.pkl'
    HR_true = './true_HR.pkl'
    TEMP_pred = './pred_TEMP.pkl'
    TEMP_true = './true_TEMP.pkl'
    l1 = load(EDA_pred)
    l2 = load(EDA_true)
    l3 = load(HR_pred)
    l4 = load(HR_true)
    l5 = load(TEMP_pred)
    l6 = load(TEMP_true)
    n = len(l1)
    if not os.path.exists('./csv'):
        os.makedirs('./csv')
    for i in range(n):
        cur_l1 = l1[i].flatten()
        cur_l2 = l2[i].values.flatten()
        cur_l3 = l3[i].flatten()
        cur_l4 = l4[i].values.flatten()
        cur_l5 = l5[i].flatten()
        cur_l6 = l6[i].values.flatten()
        curmat = np.hstack((cur_l1.reshape((-1,1)),cur_l2.reshape((-1,1)),cur_l3.reshape((-1,1)),
            cur_l4.reshape((-1,1)),cur_l5.reshape((-1,1)),cur_l6.reshape((-1,1))))
        np.savetxt(("./csv/%d.csv"% i), curmat, delimiter=",")
    
main()    