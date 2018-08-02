import numpy as np
from datetime import datetime
import pickle

##Shedding
f = open('dataExportForRelease/sqlViews/vw_shedding_release.txt', 'r')
line = []
for lines in f.readlines():
    line.append(lines.strip().split('\t'))
names = ['HRV15-002', 'HRV15-003', 'HRV15-004', 'HRV15-005', 'HRV15-006', 'HRV15-007', 'HRV15-008', 'HRV15-009', 'HRV15-011', 'HRV15-012', 'HRV15-013',  'HRV15-017', 'HRV15-018', 'HRV15-019', 'HRV15-020', 'HRV15-021', 'HRV15-022', 'HRV15-023', 'HRV15-024']


day_shelist = []
sheddinglist = []

for i in range(len(names)):
    shedding = []
    day_she = []
    for j in range (len(line)):
        if(names[i] == line[j][0]):
            day_she.append(datetime.strptime(line[j][1],'%Y-%m-%d'))
            if(line[j][3] == 'Positive'):
                shedding.append(1)
            else:
                shedding.append(0.0)
    day_shelist.append(day_she)
    sheddinglist.append(shedding)

exp_start_date = 14
exp_end_date = 22    
time_window = 5
days = list(range(exp_start_date,exp_end_date-time_window+1))

total_shedlist = np.zeros(len(names))
for i in range(len(names)):
    cur_day_she = day_shelist[i]
    total_shed = []
    for d in range(len(days)):
        tmp_shed = 0
        for j in range(len(cur_day_she)):
            if(cur_day_she[j].day>=days[d] and cur_day_she[j].day<=days[d]+(time_window-1)):
                tmp_shed = tmp_shed + sheddinglist[i][j]
        total_shed.append(tmp_shed)
    total_shedlist[i] = max(total_shed)

print(total_shedlist)

##Symptom
f = open("dataExportForRelease/sqlViews/vw_dailySymptoms_release.txt", "r")
lines = f.readlines()

line = []
for i in range(1,len(lines)):
    line.append(lines[i].split('\t'))
    
day_symlist = []
symptlist = []

for i in range(len(names)):
    day = []
    sympt = []
    for j in range(len(line)):
        tmp_sympt = 0
        if(names[i] == line[j][0]):
            day.append(datetime.strptime(line[j][1],'%Y-%m-%d'))
            for n in range(7,15):
                #print(j)
                #print(line[85][n])
                tmp_sympt = tmp_sympt + int(line[j][n])
            #print(tmp_sympt)
            sympt.append(tmp_sympt)
    day_symlist.append(day)
    symptlist.append(sympt)    

   
total_symptlist = np.zeros(len(names))
for i in range(len(names)):
    cur_day_sympt = day_shelist[i]
    total_sympt = []
    for d in range(len(days)):
        tmp_sym = 0
        for j in range(len(cur_day_sympt)):
            if(cur_day_sympt[j].day>=days[d] and cur_day_sympt[j].day<=days[d]+(time_window-1)):
                tmp_sym = tmp_sym + symptlist[i][j]
        total_sympt.append(tmp_sym)
    total_symptlist[i] = max(total_sympt)

print(total_symptlist) 

labels=np.zeros(len(names))
shed_thresh = 2
sym_thresh = 6
num_infected = 0
for i in range(len(names)):
    if(total_shedlist[i]>=shed_thresh and total_symptlist[i]>=sym_thresh):
        labels[i] = 1
        num_infected = num_infected + 1
        
print(labels)
print("infected rate:")
print(num_infected/len(labels))
pickle.dump(labels,open("labels.pkl", "wb" ))
    
    
    
    
    
    
    
    
    
    
    
    
    
    