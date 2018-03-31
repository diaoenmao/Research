import pandas as pd
import numpy as np
import pickle
from scipy import signal
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import copy
import matplotlib.pyplot as plt
from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, lfilter, firwin, freqz
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
import numpy as np
from scipy.fftpack import fft, ifft
import scipy.signal as signal
import numpy as np
import pickle
from scipy import signal
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import copy
import matplotlib.pyplot as plt
import pytz # new import
import datetime
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import preprocessing
from sklearn import utils
import copy
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.model_selection import LeaveOneOut
from sklearn import linear_model



effective_subject_size = 19
full_indicator = []
for i in range(10):
    full_indicator.append('0'+'%s'%i)
for i in range(10,60):
    full_indicator.append('%s'%i)

def read_data(path):
	df_groupby_minute_total = []
	for i in range(effective_subject_size):
	    a = pd.read_csv(path + '_%s'%i)
	    df_groupby_minute_total.append(a.ix[:,1:])
	return df_groupby_minute_total

def determine_explore_range(start_train,end_train,start_test,end_test,start_hour,end_hour):
	train_start = np.datetime64('2015-09-%sT%s:00:00.000000000'%(str(start_train),start_hour))
	train_end = np.datetime64('2015-09-%sT%s:59:00.000000000'%(str(end_train),end_hour))
	df_split_train = pd.DataFrame()
	df_split_train['time'] = [train_start,train_end]
	index = pd.DatetimeIndex(df_split_train['time'])
	index = index.astype(np.int64)// 10**9
	df_split_train['time_unix'] = index

	test_start = np.datetime64('2015-09-%sT%s:00:00.000000000'%(str(start_test),start_hour))
	test_end = np.datetime64('2015-09-%sT%s:59:00.000000000'%(str(end_test),end_hour))
	df_split_test = pd.DataFrame()
	df_split_test['time'] = [test_start,test_end]
	index = pd.DatetimeIndex(df_split_test['time'])
	index = index.astype(np.int64)// 10**9
	df_split_test['time_unix'] = index 
	return df_split_train,df_split_test

def explore_data(df_groupby_minute_total,df_split):
	df = []
	for i in range(effective_subject_size):
	    a = df_groupby_minute_total[i].loc[df_groupby_minute_total[i].TIMESTAMP.map(int) >= df_split.time_unix.values[0]].loc[df_groupby_minute_total[i].TIMESTAMP.map(int) <= df_split.time_unix.values[-1]]
	    a.index = range(len(a))
	    a['Hour'] = a.Minute.apply(lambda x: x[:13])
	    a['Min_indicator'] = a.Minute.apply(lambda x: x[14:17])
	    df.append(a)
	return df

#compensate missing data
def complete(n,df_train_test):
    df_complete = []
    for i in range(len(df_train_test[n].Hour.unique())):
        a = df_train_test[n][df_train_test[n].Hour==df_train_test[n].Hour.unique()[i]] # filter the minite based on each hour
        a.index = range(len(a))
        if len(a) != 60:
            EDA_avg,HR_avg,TEMP_avg = a.EDA.mean(),a.HR.mean(),a.TEMP.mean()
            missing = list(set(full_indicator) - set(a.Min_indicator))
            df = pd.DataFrame()
            df['Min_indicator'] = missing
            df['EDA'] = EDA_avg
            df['HR'] = HR_avg
            df['TEMP'] = TEMP_avg
            df['Minute'] = a.Minute[0][:14]
            df['Minute'] = df['Minute'] + df['Min_indicator']
            df['Hour'] = df.Minute.apply(lambda x: x[:13])
            df = pd.concat([a,df]).sort_values('Min_indicator')    
            df.index = range(len(df))
            df_complete.append(df)
        else:
            df_complete.append(a)
    return df_complete


def rf_predict(n,input_,output_):
    impo = []
    output_ = output_
    input_arr = input_
    lab_enc = preprocessing.LabelEncoder()
    encoded = lab_enc.fit_transform(output_)
    dictionary = dict(zip(encoded, output_))
    clf_rf = RandomForestClassifier(random_state=1)
    clf_rf.fit(input_arr,encoded.reshape(len(encoded),))
    impo.append(clf_rf.feature_importances_)
    lis_in = []
    for i in range(24):
        update_input = input_arr[-1][i:].tolist() + lis_in
        result = clf_rf.predict(np.asarray(update_input).reshape(1, -1))
        lis_in += [dictionary[result[0]]]
    return lis_in,impo

def input_output(df_train,para,n):
    sub_df_train = np.asanyarray(df_train[n].groupby('Hour',as_index = False).mean()[para])
    input_ = []
    for i in range(0,len(sub_df_train)-24):
        input_.append(sub_df_train[i:i+24])
    output_ = []
    for i in np.arange(24,len(sub_df_train)):
        output_.append(sub_df_train[i])
    return input_,output_

def predict_impo(df_train,para):
    impo_list = []
    pred_list = []
    for i in range(len(df_train)):
        input_,output_ = input_output(df_train,para,i)
        pred,impo = rf_predict(i,input_,output_)
        pred_list.append(pred)
        impo_list.append(impo)
    return pred_list,impo_list

def true_value(df_test,para):
    true_list = []
    for i in range(len(df_test)):
        sub_df_test = df_test[i].groupby('Hour',as_index = False).mean()[para]
        true_list.append(np.asanyarray(sub_df_test))
    return true_list


def distance_arr_(pred_EDA,pred_HR,pred_TEMP,true_TEMP,true_EDA,true_HR):
	distance_arr = np.zeros((effective_subject_size-1,3))
	index = 14
	pred_TEMP = pred_TEMP[:index] + pred_TEMP[index+1: ]
	true_TEMP = true_TEMP[:index] + true_TEMP[index+1: ]
	pred_EDA = pred_EDA[:index] + pred_EDA[index+1: ]
	true_EDA = true_EDA[:index] + true_EDA[index+1: ]
	pred_HR = pred_HR[:index] + pred_HR[index+1: ]
	true_HR = true_HR[:index] + true_HR[index+1: ]
	for i in range(effective_subject_size-1):  
		predicted_term_eda_std = np.std(pred_EDA[i])
		predicted_term_temp_std = np.std(pred_TEMP[i])
		predicted_term_hr_std = np.std(pred_HR[i])
		if predicted_term_hr_std == 0:
			predicted_term_hr_std = 0.1
		distance,_ = fastdtw(np.asarray(pred_TEMP[i]).reshape(24,),true_TEMP[i].reshape(24,),dist=euclidean)
		distance_arr[i,0] = distance/predicted_term_temp_std
		distance,_ = fastdtw(np.asarray(pred_EDA[i]).reshape(24,),true_EDA[i].reshape(24,),dist=euclidean)
		distance_arr[i,1] = distance/predicted_term_eda_std
		distance,_ = fastdtw(np.asarray(pred_HR[i]).reshape(24,),true_HR[i].reshape(24,),dist=euclidean)
		distance_arr[i,2] = distance/predicted_term_hr_std
	return distance_arr


def svm_(alpha,dtw,label,names):
	index = 14
	label = label[:index] + label[index+1 :]
	names = names[:index] + names[index+1 :]
	X = dtw
	y = np.asanyarray(label)
	loo = LeaveOneOut()
	loo.get_n_splits(X)
	count = 0
	correct_sub = []
	fp_sub = []
	fn_sub = []
	for train_index, test_index in loo.split(X):
		#print (train_index,test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
# 		scaler = StandardScaler() 
# 		scaler.fit(X_train) 
# 		X_train = scaler.transform(X_train)  
# 		X_test = scaler.transform(X_test) 
		clf = SVC(C = alpha)
		#clf = linear_model.LogisticRegression(C=alpha)
		#clf = linear_model.LogisticRegression(C=alpha)
		clf.fit(X_train, y_train)  
		Y_predict = clf.predict(X_test)
		if Y_predict == y_test:
			#correct_sub.append(names[test_index])
			count += 1
		else:
			if Y_predict == 1:
				#print (test_index)
				fp_sub.append(names[int(test_index)])
			elif Y_predict == 0:
				fn_sub.append(names[int(test_index)])
		#print (Y_predict)
	return count/(len(X)),fp_sub,fn_sub

def pick_best_fn_fp(distance_new,label,names):	
	alpha = [-8,-2,-1,0,1,2,3,4,5]
	#alpha = np.arange(0.0000001,10000,10)
	record = []
	fp_sub_list = []
	fn_sub_list = []
	for x in alpha:
	    #print (10**(x))
	    rate,fp_sub,fn_sub = svm_(10**x,distance_new,label,names)

	    record.append(rate)
	    fp_sub_list.append(fp_sub)
	    fn_sub_list.append(fn_sub)
	max_record = max(record)
	index_max = record.index(max_record)
	alpha_max = alpha[index_max]
	fp_sub_max = fp_sub_list[index_max]
	fn_sub_max = fn_sub_list[index_max]
	return max_record,alpha_max,fp_sub_max,fn_sub_max

def plot_sick(true,pred,label,names,para,indicator,n):
	index = 14
	label = label[:index] + label[index+1 :]
	names = names[:index] + names[index+1 :]
	fig = plt.figure(figsize=(20,40))
	n = 0
	for i in range(effective_subject_size-1):   
		if label[i] == 1:
			n+=1
			plt.title('%s_sick'%para)
			plt.subplot(10,1,n)
			plt.plot(pred[i],label = 'pred_original',c='y',linewidth=7.0,alpha = 0.5)
			plt.plot(true[i],label = 'true_original',c = 'g',linewidth=7.0,alpha = 0.5)
			plt.ylabel('subject_%s'%(names[i]))
			plt.legend(loc=0).draggable()

	plt.savefig('result/%s.png' % (para+'_sick'+indicator))
	plt.close(fig)

def plot_not_sick(true,pred,label,names,para,indicator,n):
	index = 14
	label = label[:index] + label[index+1 :]
	names = names[:index] + names[index+1 :]
	fig = plt.figure(figsize=(20,40))
	n = 0
	for i in range(effective_subject_size-1):   
		if label[i] == 0:
			n+=1
			plt.title('%s_not_sick'%para)
			plt.subplot(8,1,n)
			plt.plot(pred[i],label = 'pred_original',c='y',linewidth=7.0,alpha = 0.5)
			plt.plot(true[i],label = 'true_original',c = 'g',linewidth=7.0,alpha = 0.5)
			plt.ylabel('subject_%s'%(names[i]))
			plt.legend(loc=0).draggable()

	plt.savefig('result/%s.png' % (para+'_not_sick'+indicator))
	plt.close(fig)


def scatter_plot(distance_new,label,names,indicator,n):
	index = 14
	label = label[:index] + label[index+1 :]
	names = names[:index] + names[index+1 :]

	x,y,z = distance_new[:,0],distance_new[:,1],distance_new[:,2]
	fig = plt.figure(figsize=(10,5))
	ax = fig.add_subplot(111, projection='3d')
	sick = []
	not_sick = []
	sick_name = []
	not_sick_name = []
	for i in range(len(label)): 
	    if label[i] == 1:
	        sick.append(i)   
	        sick_name.append(names[i].split('-')[1])
	        #print (i)
	    else:
	        not_sick.append(i)
	        not_sick_name.append(names[i].split('-')[1])
	        #print (';%',i)

	for i in range(len(sick)):    
	        ax.text(x[sick[i]], y[sick[i]], z[sick[i]],  '%s' % (sick_name[i]), size=10, zorder=1,horizontalalignment='right', color='r')
	        
	for i in range(len(not_sick)):    
	        ax.text(x[not_sick[i]], y[not_sick[i]], z[not_sick[i]],  '%s' % (not_sick_name[i]), size=10, zorder=10, horizontalalignment='left', color='b')

	ax.scatter(x[sick], y[sick], z[sick], c='r', marker='o',label = 'sick')

	ax.scatter(x[not_sick], y[not_sick], z[not_sick], c='b', marker='o',label = 'not sick')
	ax.set_xlabel('DTW_EDA')
	plt.title('TIME_DTW')

	ax.set_ylabel('DTW_TEMP')
	ax.set_zlabel('DTW_HR')
	plt.legend(loc="center left")
	plt.tight_layout()

	#plt.savefig('result/result%s/scatter_plot_%s.png' % (str(n),indicator))
	plt.savefig('result/scatter_plot_%s.png' % (indicator))
	plt.close(fig)
	#plt.show()

def result_time_shift(path,start_train,end_train,start_test,end_test,start_hour,end_hour,names,label,n):
	df_groupby_minute_total = read_data(path)
	df_split_train,df_split_test = determine_explore_range(start_train,end_train,start_test,end_test,start_hour,end_hour)
	df_train = explore_data(df_groupby_minute_total,df_split_train)
	df_test = explore_data(df_groupby_minute_total,df_split_test)
	EDA_pred_list,EDA_impo_list = predict_impo(df_train,'EDA')
	HR_pred_list,HR_impo_list = predict_impo(df_train,'HR')
	TEMP_pred_list,TEMP_impo_list = predict_impo(df_train,'TEMP')
	EDA_true_list = true_value(df_test,'EDA')
	TEMP_true_list = true_value(df_test,'TEMP')
	HR_true_list = true_value(df_test,'HR')

	plot_sick(EDA_true_list,EDA_pred_list,label,names,'EDA',(str(start_train)+start_hour),n)
	plot_sick(HR_true_list,HR_pred_list,label,names,'HR',(str(start_train)+start_hour),n)
	plot_sick(TEMP_true_list,TEMP_pred_list,label,names,'TEMP',(str(start_train)+start_hour),n)
	plot_not_sick(EDA_true_list,EDA_pred_list,label,names,'EDA',(str(start_train)+start_hour),n)
	plot_not_sick(HR_true_list,HR_pred_list,label,names,'HR',(str(start_train)+start_hour),n)
	plot_not_sick(TEMP_true_list,TEMP_pred_list,label,names,'TEMP',(str(start_train)+start_hour),n)


	distance_new = distance_arr_(EDA_pred_list,HR_pred_list,TEMP_pred_list,TEMP_true_list,EDA_true_list,HR_true_list)
	scatter_plot(distance_new,label,names,str(start_train)+start_hour,n)
	max_record,alpha_max,fp_sub_max,fn_sub_max = pick_best_fn_fp(distance_new,label,names)
	return max_record,10**alpha_max,fp_sub_max,fn_sub_max,distance_new





