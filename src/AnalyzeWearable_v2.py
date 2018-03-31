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
ifchunkstored = True
ifdtwstored = False
ifoutput = False 
ifsynchronized = False

label = pickle.load(open("labels.pkl", "rb" )).tolist()

total_days = 8
shift_hours = 8
chunks_per_day = int(24/shift_hours)
total_chunks = int(chunks_per_day*total_days)-(chunks_per_day-1)
exp_start_date = 14
exp_end_date = 22
days = list(range(exp_start_date,exp_end_date))
hours = np.array(range(0,(chunks_per_day-1)*shift_hours+1,shift_hours))
    
# if(not ifchunkstored):
    # ##Shift Window Chunk
    # chunkslist = []
    # for i in range(effective_subject_size):
        # print(i)
        # cur_subject = subjectlist[i]
        # cur_time_sampleSize = len(cur_subject)
        # cur_id = id[i]
        # chunks=[]
        # for l in range(total_chunks):
            # chunks.append(dict(zip(extract_feature, [[] for i in range(len(extract_feature))])))
        # for d in range(len(days)):
            # for h in range(len(hours)):
                # if(not(days[d]==exp_end_date-1 and hours[h]!=hours[0])):
                    # chunk_idx = d*chunks_per_day+h
                    # out_cur_window = dict(zip(extract_feature, [[] for i in range(len(extract_feature))]))
                    # for j in range(cur_time_sampleSize):
                        # print(j)
                        # cur_time_sample = cur_subject[j]
                        # prev_out_cur_window = copy.deepcopy(out_cur_window)
                        # out_cur_window = dict(zip(extract_feature, [[] for i in range(len(extract_feature))]))                           
                        # for m in cur_time_sample:
                            # start_time = cur_time_sample[m][1]
                            # end_time = cur_time_sample[m][2]
                            # start_date = start_time.day
                            # start_hour = start_time.hour
                            # end_date = end_time.day
                            # end_hour = end_time.hour
                            # # if(days[d]==exp_end_date-1):
                                # # if(start_date>=days[d] and end_date<=days[d] and start_hour>=hours[h]):
                                    # # chunks[chunk_idx][m]=np.concatenate((chunks[chunk_idx][m],cur_time_sample[m][0]))
                            # # else:           
                            # if(hours[h]==hours[0]):
                                # if(start_date>=days[d] and end_date<=days[d] and start_hour>=hours[h]):   
                                    # chunks[chunk_idx][m]=np.concatenate((prev_out_cur_window[m],chunks[chunk_idx][m],cur_time_sample[m][0]))
                            # else:
                                # if(start_date>=days[d] and end_date<=days[d+1] and start_hour>=hours[h] and end_hour<hours[h]):
                                    # chunks[chunk_idx][m]=np.concatenate((prev_out_cur_window[m],chunks[chunk_idx][m],cur_time_sample[m][0]))
                                # elif(start_date>=days[d] and end_date<=days[d+1] and start_hour>=hours[h] and end_hour>=hours[h]):
                                    # samples_in_cur_window = ((hours[h]-start_hour)*3600-(start_time.minute)*60-start_time.second)*cur_time_sample[m][3]
                                    # in_cur_window = cur_time_sample[m][0][:samples_in_cur_window]
                                    # chunks[chunk_idx][m]=np.concatenate((prev_out_cur_window[m],chunks[chunk_idx][m],in_cur_window))
                                    # out_cur_window[m] = cur_time_sample[m][0][samples_in_cur_window:]
        # chunkslist.append(chunks)
    # print("Shift Window complete")

    # ##Short-Time Fourier Transform
    # dataSample_per_window = 4096
    # fft_chunkslist = []   
    # for i in range(effective_subject_size):
        # print(i)
        # cur_chunks = chunkslist[i]
        # fft_chunks=[]
        # for l in range(total_chunks):
            # fft_chunks.append(dict.fromkeys(extract_feature))
        # for j in range(total_chunks):
            # print(j)
            # cur_chunk = cur_chunks[j]
            # for m in cur_chunk:
                # if(len(cur_chunk[m])!=0):
                    # #print(len(cur_chunk[m]))
                    # resampled = signal.resample(cur_chunk[m], dataSample_per_window)
                    # fft_chunks[j][m]=np.abs(np.fft.fft(resampled))
        # fft_chunkslist.append(fft_chunks)    
    # print("STFT complete")
    # pickle.dump((chunkslist,fft_chunkslist),open("chunks.pkl", "wb" ))
# else:
    # chunkslist,fft_chunkslist = pickle.load(open("chunks.pkl", "rb" ))
    
    
##Short-Time Fourier Transform    
fft_size = 16384
fft_chunkslist = []
time_stamps = []
hopping_percent = 1/4   
if(not ifchunkstored):
    long_chunks_list = [] 
    for i in range(effective_subject_size):
        #print(i)
        cur_subject = subjectlist[i]
        cur_time_sampleSize = len(cur_subject)
        long_chunks = dict(zip(extract_feature, [[] for i in range(len(extract_feature))]))
        for j in range(cur_time_sampleSize):
            cur_time_sample = cur_subject[j]
            for m in cur_time_sample:
                long_chunks[m] = np.concatenate((long_chunks[m],cur_time_sample[m][0]))
        long_chunks_list.append(long_chunks)
        
    for i in range(effective_subject_size):
        cur_long_chunks = long_chunks_list[i]
        fft_chunks= dict.fromkeys(extract_feature)
        for m in cur_long_chunks:
            hz = subjectlist[i][0][m][3]
            print(m)
            window_size =  hz*3600
            overlap_size = hopping_percent*window_size
            #baseline_idx = np.floor(3*24*3600*hz/(window_size-overlap_size)).astype(np.int)
            #baseline_idx = 20
            #print(baseline_idx)
            n = 101
            a = signal.firwin(n, cutoff = 0.2, window = "hamming")
            #Spectral inversion
            a = -a
            a[int(n/2)] = a[int(n/2)] + 1
            tmp = np.convolve(cur_long_chunks[m], a)
            f,t,Zxx=signal.stft(tmp,hz,nperseg=window_size,noverlap=overlap_size,nfft=fft_size)
            Zxx=np.abs(Zxx)
            #plt.plot(Zxx[:,0],color='r')
            #plt.plot(Zxx[:,20],color='b')
            #plt.show()
            #tmp_Zxx = np.mean(Zxx[:,:baseline_idx], axis=1).reshape((-1,1))
            #print(tmp_Zxx)
            #print(Zxx)
            #t=t[baseline_idx:]
            #print(tmp_Zxx.shape)
            #print(Zxx[:,baseline_idx:].shape)
            #Zxx=np.concatenate((tmp_Zxx,Zxx),axis=1)
            time_stamps.append(Zxx.shape[1])
            fft_chunks[m] = [f,t,Zxx]
        fft_chunkslist.append(fft_chunks)
    time_stamps = np.array(time_stamps)
    print("STFT complete")
    pickle.dump((fft_chunkslist,time_stamps),open("chunks.pkl", "wb" ))
else:
   fft_chunkslist,time_stamps = pickle.load(open("chunks.pkl", "rb" ))

if(not ifdtwstored):
    ##Dynamic Time Warping
    dtwlist = []
    for i in range(effective_subject_size):
        print(i)
        cur_fft_chunks = fft_chunkslist[i]
        dtw = dict(zip(extract_feature, [[] for i in range(len(extract_feature))]))
        for m in cur_fft_chunks:
            print(m)
            for j in range(1,cur_fft_chunks[m][2].shape[1]):
                print(j)
                tmpdtw,_ = fastdtw(cur_fft_chunks[m][2][0],cur_fft_chunks[m][2][j],dist=euclidean)
                dtw[m] = np.concatenate((dtw[m],np.array([tmpdtw])))
            plt.figure()
            plt.plot(dtw[m])
            plt.title(('%s labeled as %d' % (m,label[i])))
            plt.savefig(('./result/%d_%d_%s.png' % (i,label[i],m)))
        dtwlist.append(dtw)          
    print("DTW complete")
    pickle.dump(dtwlist,open("dtw.pkl", "wb" ))
else:
    dtwlist = pickle.load(open("dtw.pkl", "rb" ))

    
# if(not ifdtwstored):
    # ##Dynamic Time Warping
    # dtwlist = []
    # baseidxlist = []
    # for i in range(effective_subject_size):
        # print(i)
        # cur_fft_chunks = fft_chunkslist[i]
        # dtw = dict(zip(extract_feature, [[] for i in range(len(extract_feature))]))
        # baseidx = dict.fromkeys(extract_feature,0)
        # #print(dtw)
        # for j in range(1,total_chunks):
            # print(j)
            # #print(len(cur_fft_chunks))
            # cur_fft_chunk = cur_fft_chunks[j]
            # #print(cur_fft_chunk)
            # for m in cur_fft_chunk:
                # for l in range(total_chunks):
                    # if(cur_fft_chunks[l][m] is not None):
                        # baseidx[m] = l
                        # break
                # if(cur_fft_chunk[m] is None or j<=baseidx[m]):
                    # dtw[m]=np.concatenate((dtw[m],np.array([0])))                  
                # else:
                    # # print(baseidx[m])
                    # # print(j)
                    # tmpdtw,_ = fastdtw(cur_fft_chunks[baseidx[m]][m],cur_fft_chunk[m],dist=euclidean)
                    # #print(cur_fft_chunk[m])
                    # dtw[m] = np.concatenate((dtw[m],np.array([tmpdtw])))
            # # zero_idx = np.array(np.where(dtw[m]==0))
            # # if(zero_idx.shape[1]!=0):
                # # dtw[m][zero_idx] = dtw[m][zero_idx+1]
            # #print(dtw)
        # dtwlist.append(dtw)
        # baseidxlist.append(baseidx)            
    # print("DTW complete")
    # pickle.dump((dtwlist,baseidxlist),open("dtw.pkl", "wb" ))
# else:
    # dtwlist,baseidxlist= pickle.load(open("dtw.pkl", "rb" ))

# print(dtwlist)
# print(baseidxlist)

## Output

# wearable = []
# for i in range(effective_subject_size):
    # cur_dtw = dtwlist[i]
    # dm = np.zeros((len(extract_feature),total_chunks-1))
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
        # dm[feature_idx,]= cur_dtw[m]/np.max(cur_dtw[m])
        # feature_idx = feature_idx + 1
    # wearable.append(dm)

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

## Synchronize with RNA
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



