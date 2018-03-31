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


def rf_predict(n,PARA,df_train,rand_state):
    df_complete = pd.concat(complete(n,df_train),axis =1)
    para_concat = df_complete[PARA]
    label_arr = []
    input_arr = []
    impo = []
    #print (para_concat.shape)
    for i in range(24):
        input_ = para_concat.values[:,i:i+24]
        label = para_concat.values[:,i+24]
        label_arr.append(label)
        input_arr.append(input_)
    
    #lab_enc = preprocessing.LabelEncoder()
    predict_result = []
    a = []
    for i in range(24):
       # print(i)
        lab_enc = preprocessing.LabelEncoder()
        output_ = label_arr[i]
        encoded = lab_enc.fit_transform(output_)
        dictionary = dict(zip(encoded, output_))
        clf_rf = RandomForestClassifier(random_state = rand_state)
        clf_rf.fit(input_arr[i],encoded.reshape(len(encoded),))
        impo.append(clf_rf.feature_importances_)
        #print (dictionary)
        if i == 0:
            lab_enc = preprocessing.LabelEncoder()
            update_input = para_concat.values[:,i+24:] 
            update_label = clf_rf.predict(update_input).reshape(len(update_input),1)
            #print (update_label[0])
            update_label = [dictionary[x[0]] for x in update_label]
        elif i == 1:
            
            update_input = np.concatenate((para_concat.values[:,i+24:],a[0].reshape(len(update_input),1)),axis = 1)
            update_label = clf_rf.predict(update_input).reshape(len(update_input),1)
            update_label = [dictionary[x[0]] for x in update_label]

        else:
            #print (np.asanyarray(a).shape)
            update_input = np.concatenate((para_concat.values[:,i+24:],np.asanyarray(a).T),axis = 1)
            update_label = clf_rf.predict(update_input).reshape(len(update_input),1)
            update_label = [dictionary[x[0]] for x in update_label]

        predict_result.append(update_label)
        a.append(np.asanyarray(update_label))
        #print (a)
        
    return impo, predict_result, a




def predict_result(PARA,df_train,rand_state):
	pred = []
	impo_list = []
	for i in range(effective_subject_size):
	    impo, predict_result,a = rf_predict(i,PARA,df_train,rand_state)
	    pred.append(np.asanyarray(predict_result))
	    impo_list.append(impo)
	return impo_list,pred

def true_result(PARA,df_test):
	true = []
	for i in range(effective_subject_size):
	    df_complete = pd.concat(complete(i,df_test),axis =1)
	    para_concat = df_complete[PARA]
	    #print(para_concat.shape[1])
	    true.append(para_concat)
	return true

def plot_comparison_sick(true,pred,label,names):
	fig = plt.figure(figsize=(20,40))
	n = 0
	for i in range(effective_subject_size):
	    if i !=14:
	        if label[i] == 1:
	            n+=1
	            plt.title('temp_sick')
	            plt.subplot(10,1,n)
	            plt.ylabel('subject_%s'%names[i])
	            plt.plot(pred[i].reshape(60*24,))
	            plt.plot(true[i].values.T.reshape(24*60,))
	plt.show()

def plot_comparison_not_sick(true,pred,label,names):
	fig = plt.figure(figsize=(20,40))
	n = 0
	for i in range(effective_subject_size):
	    if i !=14:
	        if label[i] == 0:
	            n+=1
	            plt.title('temp_not_sick')
	            plt.subplot(8,1,n)
	            plt.ylabel('subject_%s'%names[i])
	            plt.plot(pred[i].reshape(60*24,))
	            plt.plot(true[i].values.T.reshape(24*60,))
	plt.show()

def distance_arr_(pred_EDA,pred_HR,pred_TEMP,true_TEMP,true_EDA,true_HR):
	distance_arr = np.zeros((effective_subject_size,3))
	for i in range(effective_subject_size):
	    if i not in [14]:
	        distance,_ = fastdtw(pred_TEMP[i].T.reshape(60*24,),true_TEMP[i].values.reshape(60*24,),dist=euclidean)
	        distance_arr[i,0] = distance
	        distance,_ = fastdtw(pred_EDA[i].T.reshape(60*24,),true_EDA[i].values.reshape(60*24,),dist=euclidean)
	        distance_arr[i,1] = distance
	        distance,_ = fastdtw(pred_HR[i].T.reshape(60*24,),true_HR[i].values.reshape(60*24,),dist=euclidean)
	        distance_arr[i,2] = distance
	return distance_arr

def distance_arr_type2(pred_EDA,pred_HR,pred_TEMP,true_TEMP,true_EDA,true_HR):
	# remove subject 14
	distance_arr = np.zeros((effective_subject_size-1,3))
	for i in range(effective_subject_size-1):
		distance,_ = fastdtw(pred_TEMP[i].T.reshape(60*24,),true_TEMP[i].reshape(60*24,),dist=euclidean)
		distance_arr[i,0] = distance
		distance,_ = fastdtw(pred_EDA[i].T.reshape(60*24,),true_EDA[i].reshape(60*24,),dist=euclidean)
		distance_arr[i,1] = distance
		distance,_ = fastdtw(pred_HR[i].T.reshape(60*24,),true_HR[i].reshape(60*24,),dist=euclidean)
		distance_arr[i,2] = distance
	return distance_arr



# Low pass filter

def lpf_plot_sick(true,pred,label,names,thre,para,indicator,n):
	fig = plt.figure(figsize=(20,40))
	n = 0
	for i in range(effective_subject_size):
	    if i !=14:
	        if label[i] == 1:
	            n+=1
	            plt.title('%s_sick'%para)
	            plt.subplot(10,1,n)
	            #plt.ylabel('subject_%s'%names[i])
	            plt.plot(pred[i].reshape(60*24,),label = 'pred_original',c='y',linewidth=7.0,alpha = 0.5)
	            plt.plot(true[i].values.T.reshape(24*60,),label = 'true_original',c = 'g',linewidth=7.0,alpha = 0.5)
	            rft = np.fft.rfft(pred[i].reshape(60*24,))
	            rft[thre:] = 0   
	            y_smooth = np.fft.irfft(rft)
                
	            plt.plot(y_smooth.reshape(60*24,), label='pred_smoothed',c = 'k',alpha = 1)
	            rft = np.fft.rfft(true[i].values.T.reshape(60*24,))
	            rft[thre:] = 0   
	            y_smooth = np.fft.irfft(rft)
	            plt.plot(y_smooth.reshape(60*24,), label='true_smoothed',c = 'r',alpha = 1)
	            plt.ylabel('subject_%s_threshold value = %d'%(names[i],thre))
	            plt.legend(loc=0).draggable()
	#plt.savefig('result/result%s/%s.png' % (str(n),para+'_sick'+indicator))
	plt.savefig('result/%s.png' % (para+'_sick'+indicator))
	plt.close(fig)


def lpf_plot_not_sick(true,pred,label,names,thre,para,indicator,n):
	fig = plt.figure(figsize=(20,40))
	n = 0
	for i in range(effective_subject_size):
	    if i !=14:
	        if label[i] == 0:
	            n+=1
	            plt.title('%s_not_sick'%para)
	            plt.subplot(8,1,n)
	            plt.ylabel('subject_%s'%names[i])
	            plt.plot(pred[i].reshape(60*24,),label = 'pred_original',c='y',linewidth=7.0,alpha = 0.5)
	            plt.plot(true[i].values.T.reshape(24*60,),label = 'true_original',c = 'g',linewidth=7.0,alpha = 0.5)
	            rft = np.fft.rfft(pred[i].reshape(60*24,))
	            rft[thre:] = 0   
	            y_smooth = np.fft.irfft(rft)
	            plt.plot(y_smooth.reshape(60*24,), label='pred_smoothed',c = 'k',alpha = 1)
	            rft = np.fft.rfft(true[i].values.T.reshape(60*24,))
	            rft[thre:] = 0   
	            y_smooth = np.fft.irfft(rft)
	            plt.plot(y_smooth.reshape(60*24,), label='true_smoothed',c = 'r',alpha = 1)
	            plt.ylabel('subject_%s_threshold value = %d'%(names[i],thre))
	            plt.legend(loc=0).draggable()
	            
	plt.savefig('result/%s.png' % (para+'_not_sick'+indicator))
	#plt.savefig('result/result%s/%s.png' %(str(n),(para+'_not_sick'+indicator)))
	plt.close(fig)


def lpf_(y,thre):
    #plt.plot(y)
    rft = np.fft.rfft(y)
    rft[thre:] = 0   
    y_smooth = np.fft.irfft(rft)
    return y_smooth

def hpf_(y,thre):
    #plt.plot(y)
    rft = np.fft.rfft(y)
    rft[:thre] = 0   
    y_smooth = np.fft.irfft(rft)
    return y_smooth

def lpf_result(thre,pred_EDA,pred_HR,pred_TEMP,true_TEMP,true_EDA,true_HR):
	true_HR_filter = []
	pred_HR_filter = []
	true_EDA_filter = []
	pred_EDA_filter = []
	true_TEMP_filter = []
	pred_TEMP_filter = []
	for i in range(19):
	    if i != 14:
	        pred_HR_filter.append(lpf_(pred_HR[i].reshape(60*24,),thre))
	        true_HR_filter.append(lpf_(true_HR[i].values.T.reshape(60*24,),thre))
	        pred_EDA_filter.append(lpf_(pred_EDA[i].reshape(60*24,),thre))
	        true_EDA_filter.append(lpf_(true_EDA[i].values.T.reshape(60*24,),thre))
	        pred_TEMP_filter.append(lpf_(pred_TEMP[i].reshape(60*24,),thre))
	        true_TEMP_filter.append(lpf_(true_TEMP[i].values.T.reshape(60*24,),thre))
	distance_new = distance_arr_type2(pred_TEMP_filter,pred_EDA_filter,pred_HR_filter,true_TEMP_filter,true_EDA_filter,true_HR_filter)
	return distance_new

def hpf_result(thre,pred_EDA,pred_HR,pred_TEMP,true_TEMP,true_EDA,true_HR):
	true_HR_filter = []
	pred_HR_filter = []
	true_EDA_filter = []
	pred_EDA_filter = []
	true_TEMP_filter = []
	pred_TEMP_filter = []
	for i in range(19):
	    if i != 14:
	        pred_HR_filter.append(hpf_(pred_HR[i].reshape(60*24,),thre))
	        true_HR_filter.append(hpf_(true_HR[i].values.T.reshape(60*24,),thre))
	        pred_EDA_filter.append(hpf_(pred_EDA[i].reshape(60*24,),thre))
	        true_EDA_filter.append(hpf_(true_EDA[i].values.T.reshape(60*24,),thre))
	        pred_TEMP_filter.append(hpf_(pred_TEMP[i].reshape(60*24,),thre))
	        true_TEMP_filter.append(hpf_(true_TEMP[i].values.T.reshape(60*24,),thre))
	distance_new = distance_arr_type2(pred_TEMP_filter,pred_EDA_filter,pred_HR_filter,true_TEMP_filter,true_EDA_filter,true_HR_filter)
	return distance_new


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
		scaler = StandardScaler() 
		scaler.fit(X_train) 
		X_train = scaler.transform(X_train)  
		X_test = scaler.transform(X_test) 
		#clf = SVC(C = alpha)
		clf = svm.SVC(kernel='rbf',gamma = alpha)
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
	#alpha = [-2,-1,0,1,2,3,4,5]
	alpha = np.arange(0.00001,1000,10)
	record = []
	fp_sub_list = []
	fn_sub_list = []
	for x in alpha:
	    #print (10**(x))
	    rate,fp_sub,fn_sub = svm_(x,distance_new,label,names)

	    record.append(rate)
	    fp_sub_list.append(fp_sub)
	    fn_sub_list.append(fn_sub)
	max_record = max(record)
	index_max = record.index(max_record)
	alpha_max = alpha[index_max]
	fp_sub_max = fp_sub_list[index_max]
	fn_sub_max = fn_sub_list[index_max]
	return max_record,alpha_max,fp_sub_max,fn_sub_max



def result_time_shift(path,start_train,end_train,start_test,end_test,start_hour,end_hour,thre,names,label,n,rand_state):
    df_groupby_minute_total = read_data(path)
    df_split_train,df_split_test = determine_explore_range(start_train,end_train,start_test,end_test,start_hour,end_hour)

    df_train = explore_data(df_groupby_minute_total,df_split_train)

    impo_EDA,pred_EDA = predict_result('EDA',df_train,rand_state)
    impo_HR,pred_HR = predict_result('HR',df_train,rand_state)
    impo_TEMP,pred_TEMP = predict_result('TEMP',df_train,rand_state)
    df_test = explore_data(df_groupby_minute_total,df_split_test)

    true_EDA = true_result('EDA',df_test)
    true_HR = true_result('HR',df_test)
    true_TEMP = true_result('TEMP',df_test)
    distance_arr = distance_arr_(pred_EDA,pred_HR,pred_TEMP,true_TEMP,true_EDA,true_HR)
    distance_new = hpf_result(thre,pred_EDA,pred_HR,pred_TEMP,true_TEMP,true_EDA,true_HR)
    scatter_plot(distance_arr,label,names,(str(start_train)+start_hour)+'original',n)
    scatter_plot(distance_new,label,names,(str(start_train)+start_hour)+'after hpf',n)

    lpf_plot_sick(true_EDA,pred_EDA,label,names,thre,'EDA',(str(start_train)+start_hour),n)
    lpf_plot_sick(true_HR,pred_HR,label,names,thre,'HR',(str(start_train)+start_hour),n)
    lpf_plot_sick(true_TEMP,pred_TEMP,label,names,thre,'TEMP',(str(start_train)+start_hour),n)

    lpf_plot_not_sick(true_EDA,pred_EDA,label,names,thre,'EDA',(str(start_train)+start_hour),n)
    lpf_plot_not_sick(true_HR,pred_HR,label,names,thre,'HR',(str(start_train)+start_hour),n)
    lpf_plot_not_sick(true_TEMP,pred_TEMP,label,names,thre,'TEMP',(str(start_train)+start_hour),n)
    max_record,alpha_max,fp_sub_max,fn_sub_max = pick_best_fn_fp(distance_new,label,names)
    return max_record,alpha_max,fp_sub_max,fn_sub_max,distance_arr,distance_new


    