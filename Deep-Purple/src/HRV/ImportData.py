import numpy as np
import pickle
import h5py
from datetime import datetime

f = h5py.File('dataExportForRelease/wearableDevice/20160503_BIOCHRON_E4.hdf5',  "r")
f_list = list(f.keys())
finger_print = []
time_stamp_wearable = []
id = []
subjectlist = []
omit_subjectlist = []
missing_subjectlist = []
missing_time_stampslist = []
missing_featurelist = []
effective_experiments = 10
extract_feature = ["EDA","HR","TEMP"]
attributes = ["start_time","hz"]
print("Total subject size:")
print(len(f_list))
for i in range(len(f_list)):
    time_sampleSize = list(f[f_list[i]])
    timelist = []
    cur_time_stamps = []
    cur_id = f_list[i]
    if(len(time_sampleSize)>10):
        finger_print.append(time_sampleSize)
        for j in range(len(time_sampleSize)):
            #print(j)
            tmp_time = time_sampleSize[j][7:]
            cur_time_stamp = datetime.strptime(tmp_time, '%y%m%d-%H%M%S')
            cur_time_stamps.append(cur_time_stamp)
            cur_hdf5=f[f_list[i]][time_sampleSize[j]]
            cur_hdf5_feature = list(cur_hdf5)
            featurelist = dict.fromkeys(extract_feature)
            for m in range(len(cur_hdf5_feature)):
                if(not cur_hdf5_feature[m] in extract_feature):
                    continue
                #print(cur_hdf5_feature[m])
                feature = np.zeros(len(cur_hdf5[cur_hdf5_feature[m]]))
                #tmp_x = np.zeros(len(cur_hdf5[cur_hdf5_feature[m]]))
                #tmp_y = np.zeros(len(cur_hdf5[cur_hdf5_feature[m]]))
                #tmp_z = np.zeros(len(cur_hdf5[cur_hdf5_feature[m]]))
                if(cur_hdf5_feature[m] == "ACC"):
                    #print(cur_hdf5[cur_hdf5_feature[m]][:])
                    #print(cur_hdf5[cur_hdf5_feature[m]][:].astype(np.float).reshape((len(cur_hdf5[cur_hdf5_feature[m]])/3,3)))
                    tmp_x = cur_hdf5[cur_hdf5_feature[m]][:][0].astype(np.float)
                    tmp_y = cur_hdf5[cur_hdf5_feature[m]][:][1].astype(np.float)
                    tmp_z = cur_hdf5[cur_hdf5_feature[m]][:][2].astype(np.float)
                    feature=np.sqrt(tmp_x**2+tmp_y**2+tmp_z**2)
                    #print(tmp_x)
                    featurelist["ACC"]=feature
                else:     
                    if(len(cur_hdf5[cur_hdf5_feature[m]][:])!=0):
                        feature = cur_hdf5[cur_hdf5_feature[m]][:].astype(np.float)
                        start_time = datetime.fromtimestamp(cur_hdf5[cur_hdf5_feature[m]].attrs[('starttime_%s'% cur_hdf5_feature[m].lower())])
                        hz = cur_hdf5[cur_hdf5_feature[m]].attrs[('hz_%s'% cur_hdf5_feature[m].lower())]
                        duration = len(cur_hdf5[cur_hdf5_feature[m]][:])/hz
                        end_time = datetime.fromtimestamp(duration + cur_hdf5[cur_hdf5_feature[m]].attrs[('starttime_%s'% cur_hdf5_feature[m].lower())])
                        # print(start_time)
                        # print(hz)
                        # print(end_time)
                    else:
                        missing_subjectlist.append(cur_id)
                        missing_time_stampslist.append(cur_time_stamps[j])
                        missing_featurelist.append(cur_hdf5_feature[m])
                    #print(feature)
                    featurelist[cur_hdf5_feature[m]]=[feature,start_time,end_time,hz,duration]
            #print(featurelist)   
            timelist.append(featurelist)
        id.append(cur_id)
        time_stamp_wearable.append(cur_time_stamps)
        subjectlist.append(timelist)
    else:
        omit_subjectlist.append(cur_id)
output = {'id':id,'time_stamp_wearable':time_stamp_wearable,'subjectlist':subjectlist,'omit_subjectlist':omit_subjectlist,\
                    'missing_subjectlist':missing_subjectlist,'missing_time_stampslist':missing_time_stampslist,'missing_featurelist':missing_featurelist,'extract_feature':extract_feature}
pickle.dump(output, open( "data.pkl", "wb" ))
#print(id)
# print(time_stamp_wearable)
# print(finger_print)
print(omit_subjectlist)
# print(missing_subjectlist)
# print(missing_time_stampslist)
# print(missing_featurelist)
print("Effective subject size:")
print(len(finger_print))