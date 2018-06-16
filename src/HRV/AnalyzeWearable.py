import numpy as np
import pickle
from scipy import signal
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import copy
import matplotlib.pyplot as plt


output = pickle.load(open("data.pkl", "rb" ))
subjectlist = output["subjectlist"]
id = output["id"]
time_stamp_wearable = output["time_stamp_wearable"]
missing_subjectlist = output["missing_subjectlist"]
missing_time_stampslist = output["missing_time_stampslist"]
missing_featurelist = output["missing_featurelist"]
effective_subject_size = len(subjectlist)
extract_feature = output["extract_feature"]
#[feature,start_time,end_time,hz,duration]
#["EDA","HR","TEMP"]
ifchunkstored = False
ifdtwstored = False
ifoutput = False 
ifsynchronized = False
    
label = pickle.load(open("labels.pkl", "rb" )).tolist()
    
##Short-Time Fourier Transform    
fft_size = 16384
fftlist = [] 
hopping_percent = 1/4   
if(not ifchunkstored):
    for i in range(effective_subject_size):
        print(i)
        cur_subject = subjectlist[i]
        cur_time_sampleSize = len(cur_subject)
        freq_sample = dict(zip(extract_feature, [[] for i in range(len(extract_feature))]))
        valid_time_stamp = dict(zip(extract_feature, [[] for i in range(len(extract_feature))]))
        head_track = dict(zip(extract_feature, [0 for i in range(len(extract_feature))]))
        for j in range(cur_time_sampleSize):
            print(j)
            cur_time_sample = cur_subject[j]
            for m in cur_time_sample:
                hz = cur_time_sample[m][3]
                window_size =  hz*3600
                overlap_size = hopping_percent*window_size
                if(len(cur_time_sample[m][0])>window_size):
                    # print('a')
                    # print(m)
                    # print(len(cur_time_sample[m][0]))
                    # print(hz)
                    # print(cur_time_sample[m][1])
                    # print(cur_time_sample[m][2])
                    n = 101
                    a = signal.firwin(n, cutoff = 0.2, window = "hamming")
                    #Spectral inversion
                    a = -a
                    a[int(n/2)] = a[int(n/2)] + 1
                    tmp = np.convolve(cur_time_sample[m][0], a)
                    f,t,Zxx = signal.stft(tmp,hz,nperseg=window_size,noverlap=overlap_size,nfft=fft_size)
                    Zxx=np.abs(Zxx)
                    Zxx = Zxx.T.tolist()
                    head_track[m].append(head_track[m][-1]+len(Zxx))
                    # plt.plot(Zxx[0])
                    # plt.show()
                    freq_sample[m].extend(Zxx)
                    valid_time_stamp[m].append(j)
                else:
                    print('b')
                    print(m)
                    print(len(cur_time_sample[m][0]))
                    print(hz)
                    print(cur_time_sample[m][1])
                    print(cur_time_sample[m][2])
                    # Zxx = np.fft.fft(cur_time_sample[m][0],fft_size,norm='ortho')
                    # Zxx=np.abs(Zxx)
                    # Zxx = Zxx[0:int(fft_size/2)]
                    # plt.plot(Zxx)
                    # plt.show()
        del head_track[-1]
        fftlist.append(freq_sample)        
    print("STFT complete")
    pickle.dump(fftlist,open("fft.pkl", "wb" ))
else:
   fftlist = pickle.load(open("fft.pkl", "rb" ))

if(not ifdtwstored):
    ##Dynamic Time Warping
    dtwlist = []
    for i in range(effective_subject_size):
        print(i)
        cur_fft = fftlist[i]
        dtw = dict(zip(extract_feature, [[] for i in range(len(extract_feature))]))
        for m in cur_fft:
            print(m)
            for j in range(0,len(cur_fft[m])-1):
                print(j)
                #print(cur_fft[m][j])
                #print(cur_fft[m][j+1])
                tmpdtw,_ = fastdtw(cur_fft[m][j],cur_fft[m][j+1],dist=euclidean)
                print(tmpdtw)
                dtw[m] = np.concatenate((dtw[m],np.array([tmpdtw])))
        print(label)
        plt.plot(dtw['EDA'])
        plt.show()
        plt.plot(dtw['HR'])
        plt.show()
        plt.plot(dtw['TEMP'])
        plt.show()
        dtwlist.append(dtw)          
    print("DTW complete")
    pickle.dump(dtwlist,open("dtw.pkl", "wb" ))
else:
    dtwlist = pickle.load(open("dtw.pkl", "rb" ))


## Wearable output
# if(not ifoutput):
    # max_time_stamps = np.int(8*24*3600/((1-hopping_percent)*3600))
    # #max_time_stamps_idx = np.argmax(time_stamps)
    
    # wearable = []
    # wearable_feature_name = []
    # for i in range(effective_subject_size):
        # cur_dtw = dtwlist[i]
        # dm = np.zeros((len(extract_feature),max_time_stamps-1))
        # name = []
        # feature_idx = 0
        # for m in cur_dtw:
            # #print(cur_dtw[m])
            # zero_idx = np.array(np.where(cur_dtw[m]==0))
            # if(zero_idx.shape[1]!=0):
                # for j in range(zero_idx.shape[1]):
                    # idx = zero_idx[0,j]
                    # if(idx==0):
                        # cur_dtw[m][0] = cur_dtw[m][1] + 10 * np.random.randn()
                    # elif(idx==total_chunks-2):
                        # cur_dtw[m][idx] = cur_dtw[m][idx-1]
                    # else:
                        # cur_dtw[m][idx] = (cur_dtw[m][idx-1]+cur_dtw[m][idx+1])/2
            # dm[feature_idx,]=signal.resample(cur_dtw[m],max_time_stamps-1)
            # dm[feature_idx,]= dm[feature_idx,]/np.max(dm[feature_idx,])
            # feature_idx = feature_idx + 1
            # name.append(m)
        # wearable.append(dm)
        # wearable_feature_name.append(name)
    # pickle.dump((wearable,max_time_stamps,wearable_feature_name),open("wearable.pkl", "wb" ))
    # print("Wearable output complete")
# else:
    # wearable,max_time_stamps,wearable_feature_name = pickle.load(open("wearable.pkl", "rb" ))

# ## Synchronize with RNA
# length_RNA_feature=1
# feature_index = [0,1,2,3]
# if(not ifsynchronized):
    # RNA_feature = pickle.load(open("RNAdata_API.pkl", "rb" ))
    # feature = []
    # for i in range(len(wearable)):
        # feature_element = np.zeros((len(extract_feature)+1,wearable[0].shape[1]))
        # feature_element[0:3,] = wearable[i]
        # feature_element[-1,] = signal.resample(RNA_feature[i],max_time_stamps-1)
        # feature.append(feature_element[feature_index,:])
    # pickle.dump((feature,max_time_stamps,wearable_feature_name),open("feature.pkl", "wb" ))
    # print("Synchronize with RNA complete")
# else:
    # feature,max_time_stamps,wearable_feature_name = pickle.load(open("feature.pkl", "rb" ))



