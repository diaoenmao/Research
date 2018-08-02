# # # Pairwise Distance in Frequency Domain of Sub006 (no shedding and no symptom)

# # In[68]:


# distance_006_0_acc_mag, path_006_0_acc_mag = fastdtw(x = Y_006_1415_acc_mag, y = Y_006_1516_acc_mag, dist=euclidean)
# distance_006_1_acc_mag, path_006_1_acc_mag = fastdtw(x = Y_006_1415_acc_mag, y = Y_006_1617_acc_mag, dist=euclidean)
# distance_006_2_acc_mag, path_006_2_acc_mag = fastdtw(x = Y_006_1415_acc_mag, y = Y_006_1718_acc_mag, dist=euclidean)
# distance_006_3_acc_mag, path_006_3_acc_mag = fastdtw(x = Y_006_1415_acc_mag, y = Y_006_1819_acc_mag, dist=euclidean)
# distance_006_4_acc_mag, path_006_4_acc_mag = fastdtw(x = Y_006_1415_acc_mag, y = Y_006_1920_acc_mag, dist=euclidean)

# distance_006_0_hr, path_006_0_hr = fastdtw(x = Y_006_1415_hr, y = Y_006_1516_hr, dist=euclidean)
# distance_006_1_hr, path_006_1_hr = fastdtw(x = Y_006_1415_hr, y = Y_006_1617_hr, dist=euclidean)
# distance_006_2_hr, path_006_2_hr = fastdtw(x = Y_006_1415_hr, y = Y_006_1718_hr, dist=euclidean)
# distance_006_3_hr, path_006_3_hr = fastdtw(x = Y_006_1415_hr, y = Y_006_1819_hr, dist=euclidean)
# distance_006_4_hr, path_006_4_hr = fastdtw(x = Y_006_1415_hr, y = Y_006_1920_hr, dist=euclidean)

# distance_006_0_eda, path_006_0_eda = fastdtw(x = Y_006_1415_eda, y = Y_006_1516_eda, dist=euclidean)
# distance_006_1_eda, path_006_1_eda = fastdtw(x = Y_006_1415_eda, y = Y_006_1617_eda, dist=euclidean)
# distance_006_2_eda, path_006_2_eda = fastdtw(x = Y_006_1415_eda, y = Y_006_1718_eda, dist=euclidean)
# distance_006_3_eda, path_006_3_eda = fastdtw(x = Y_006_1415_eda, y = Y_006_1819_eda, dist=euclidean)
# distance_006_4_eda, path_006_4_eda = fastdtw(x = Y_006_1415_eda, y = Y_006_1920_eda, dist=euclidean)

# distance_006_0_temp, path_006_0_temp = fastdtw(x = Y_006_1415_temp, y = Y_006_1516_temp, dist=euclidean)
# distance_006_1_temp, path_006_1_temp = fastdtw(x = Y_006_1415_temp, y = Y_006_1617_temp, dist=euclidean)
# distance_006_2_temp, path_006_2_temp = fastdtw(x = Y_006_1415_temp, y = Y_006_1718_temp, dist=euclidean)
# distance_006_3_temp, path_006_3_temp = fastdtw(x = Y_006_1415_temp, y = Y_006_1819_temp, dist=euclidean)
# distance_006_4_temp, path_006_4_temp = fastdtw(x = Y_006_1415_temp, y = Y_006_1920_temp, dist=euclidean)

# distance_006_temp = [distance_006_0_temp,distance_006_1_temp,distance_006_2_temp,distance_006_3_temp,distance_006_4_temp]
# distance_006_eda = [distance_006_0_eda,distance_006_1_eda,distance_006_2_eda,distance_006_3_eda,distance_006_4_eda]
# distance_006_hr = [distance_006_0_hr,distance_006_1_hr,distance_006_2_hr,distance_006_3_hr,distance_006_4_hr]
# distance_006_acc_mag = [distance_006_0_acc_mag,distance_006_1_acc_mag,distance_006_2_acc_mag,distance_006_3_acc_mag,distance_006_4_acc_mag]

# plt.figure(figsize=(20,5))
# plt.subplot(141)
# plt.plot(distance_006_temp)
# plt.ylabel('006_distance')
# plt.title('Temp')
# plt.ylim(0,15000)
# plt.subplot(142)
# plt.title('eda')
# plt.plot(distance_006_eda)
# plt.ylim(0,30000)
# plt.subplot(143)
# plt.title('hr')
# plt.plot(distance_006_hr)
# plt.ylim(0,50000)
# plt.subplot(144)
# plt.title('acc_mag')
# plt.ylim(0,10000)
# plt.plot(distance_006_acc_mag)


# # In[55]:

# plt.figure(figsize=(20,5))

# plt.subplot(141)
# plt.plot(distance_006_temp)
# plt.ylabel('006_distance')
# plt.title('Temp')
# plt.subplot(142)
# plt.title('eda')
# plt.plot(distance_006_eda)
# plt.subplot(143)
# plt.title('hr')
# plt.plot(distance_006_hr)
# plt.subplot(144)
# plt.title('acc_mag')
# plt.plot(distance_006_acc_mag)


# # # Frequency Domain of Sub024 (no shedding and no symptom)

# # In[62]:

# # hr
# plt.figure(figsize=(20,30))

# plt.subplot(711)
# plt.title('Mag of hr of Sub 024 in Frequency Domain')
# ab_nm_024_1415_hr,ab_024_1415_hr,Y_024_1415_hr = plot_2day_square(hr_0914_024,hr_0915_024,Fs)
# plt.ylabel('DFT of 024_hr_091415')

# plt.subplot(712)
# ab_nm_024_1516_hr,ab_024_1516_hr,Y_024_1516_hr = plot_2day_square(hr_0915_024,hr_0916_024,Fs)
# plt.ylabel('DFT of 024_hr_091516')

# plt.subplot(713)
# ab_nm_024_1617_hr,ab_024_1617_hr,Y_024_1617_hr = plot_2day_square(hr_0916_024,hr_0917_024,Fs)
# plt.ylabel('DFT of 024_hr_091617')

# plt.subplot(714)
# ab_nm_024_1718_hr,ab_024_1718_hr,Y_024_1718_hr = plot_2day_square(hr_0917_024,hr_0918_024,Fs)
# plt.ylabel('DFT of 024_hr_091718')

# plt.subplot(715)
# ab_nm_024_1819_hr,ab_024_1819_hr,Y_024_1819_hr = plot_2day_square(hr_0918_024,hr_0919_024,Fs)
# plt.ylabel('DFT of 024_hr_091819')

# plt.subplot(716)
# ab_nm_024_1920_hr,ab_024_1920_hr,Y_024_1920_hr = plot_2day_square(hr_0919_024,hr_0920_024,Fs)
# plt.ylabel('DFT of 024_hr_091920')



# # eda
# plt.figure(figsize=(20,30))

# plt.subplot(711)
# plt.title('Mag of eda of Sub 024 in Frequency Domain')
# ab_nm_024_1415_eda,ab_024_1415_eda,Y_024_1415_eda = plot_2day_square(eda_0914_024,eda_0915_024,Fs)
# plt.ylabel('DFT of 024_eda_091415')

# plt.subplot(712)
# ab_nm_024_1516_eda,ab_024_1516_eda,Y_024_1516_eda = plot_2day_square(eda_0915_024,eda_0916_024,Fs)
# plt.ylabel('DFT of 024_eda_091516')

# plt.subplot(713)
# ab_nm_024_1617_eda,ab_024_1617_eda,Y_024_1617_eda = plot_2day_square(eda_0916_024,eda_0917_024,Fs)
# plt.ylabel('DFT of 024_eda_091617')

# plt.subplot(714)
# ab_nm_024_1718_eda,ab_024_1718_eda,Y_024_1718_eda = plot_2day_square(eda_0917_024,eda_0918_024,Fs)
# plt.ylabel('DFT of 024_eda_091718')

# plt.subplot(715)
# ab_nm_024_1819_eda,ab_024_1819_eda,Y_024_1819_eda = plot_2day_square(eda_0918_024,eda_0919_024,Fs)
# plt.ylabel('DFT of 024_eda_091819')

# plt.subplot(716)
# ab_nm_024_1920_eda,ab_024_1920_eda,Y_024_1920_eda = plot_2day_square(eda_0919_024,eda_0920_024,Fs)
# plt.ylabel('DFT of 024_eda_091920')



# # temp
# plt.figure(figsize=(20,30))

# plt.subplot(711)
# plt.title('Mag of temp of Sub 024 in Frequency Domain')
# ab_nm_024_1415_temp,ab_024_1415_temp,Y_024_1415_temp = plot_2day_square(temp_0914_024,temp_0915_024,Fs)
# plt.ylabel('DFT of 024_temp_091415')

# plt.subplot(712)
# ab_nm_024_1516_temp,ab_024_1516_temp,Y_024_1516_temp = plot_2day_square(temp_0915_024,temp_0916_024,Fs)
# plt.ylabel('DFT of 024_temp_091516')

# plt.subplot(713)
# ab_nm_024_1617_temp,ab_024_1617_temp,Y_024_1617_temp = plot_2day_square(temp_0916_024,temp_0917_024,Fs)
# plt.ylabel('DFT of 024_temp_091617')

# plt.subplot(714)
# ab_nm_024_1718_temp,ab_024_1718_temp,Y_024_1718_temp = plot_2day_square(temp_0917_024,temp_0918_024,Fs)
# plt.ylabel('DFT of 024_temp_091718')

# plt.subplot(715)
# ab_nm_024_1819_temp,ab_024_1819_temp,Y_024_1819_temp = plot_2day_square(temp_0918_024,temp_0919_024,Fs)
# plt.ylabel('DFT of 024_temp_091819')

# plt.subplot(716)
# ab_nm_024_1920_temp,ab_024_1920_temp,Y_024_1920_temp = plot_2day_square(temp_0919_024,temp_0920_024,Fs)
# plt.ylabel('DFT of 024_temp_091920')



# # acc_mag
# plt.figure(figsize=(20,30))
# Fs = 50
# plt.subplot(711)
# plt.title('Mag of acc_mag of Sub 024 in Frequency Domain')
# ab_nm_024_1415_mag,ab_024_1415_mag,Y_024_1415_acc_mag = plot_2day_square(acc_mag_0914_024,acc_mag_0915_024,Fs)
# plt.ylabel('DFT of 024_acc_mag_091415')

# plt.subplot(712)
# ab_nm_024_1516_acc_mag,ab_024_1516_acc_mag,Y_024_1516_acc_mag = plot_2day_square(acc_mag_0915_024,acc_mag_0916_024,Fs)
# plt.ylabel('DFT of 024_acc_mag_091516')

# plt.subplot(713)
# ab_nm_024_1617_acc_mag,ab_024_1617_acc_mag,Y_024_1617_acc_mag = plot_2day_square(acc_mag_0916_024,acc_mag_0917_024,Fs)
# plt.ylabel('DFT of 024_acc_mag_091617')

# plt.subplot(714)
# ab_nm_024_1718_acc_mag,ab_024_1718_acc_mag,Y_024_1718_acc_mag = plot_2day_square(acc_mag_0917_024,acc_mag_0918_024,Fs)
# plt.ylabel('DFT of 024_acc_mag_091718')

# plt.subplot(715)
# ab_nm_024_1819_acc_mag,ab_024_1819_acc_mag,Y_024_1819_acc_mag = plot_2day_square(acc_mag_0918_024,acc_mag_0919_024,Fs)
# plt.ylabel('DFT of 024_acc_mag_091819')

# plt.subplot(716)
# ab_nm_024_1920_acc_mag,ab_024_1920_acc_mag,Y_024_1920_acc_mag = plot_2day_square(acc_mag_0919_024,acc_mag_0920_024,Fs)
# plt.ylabel('DFT of 024_acc_mag_091920')










# # # Pairwise Distance in Frequency Domain of Sub024 (no shedding and no symptom)

# # In[63]:

# from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw

# distance_024_0_acc_mag, path_024_0_acc_mag = fastdtw(x = Y_024_1415_acc_mag, y = Y_024_1516_acc_mag, dist=euclidean)
# distance_024_1_acc_mag, path_024_1_acc_mag = fastdtw(x = Y_024_1415_acc_mag, y = Y_024_1617_acc_mag, dist=euclidean)
# distance_024_2_acc_mag, path_024_2_acc_mag = fastdtw(x = Y_024_1415_acc_mag, y = Y_024_1718_acc_mag, dist=euclidean)
# distance_024_3_acc_mag, path_024_3_acc_mag = fastdtw(x = Y_024_1415_acc_mag, y = Y_024_1819_acc_mag, dist=euclidean)
# distance_024_4_acc_mag, path_024_4_acc_mag = fastdtw(x = Y_024_1415_acc_mag, y = Y_024_1920_acc_mag, dist=euclidean)

# distance_024_0_hr, path_024_0_hr = fastdtw(x = Y_024_1415_hr, y = Y_024_1516_hr, dist=euclidean)
# distance_024_1_hr, path_024_1_hr = fastdtw(x = Y_024_1415_hr, y = Y_024_1617_hr, dist=euclidean)
# distance_024_2_hr, path_024_2_hr = fastdtw(x = Y_024_1415_hr, y = Y_024_1718_hr, dist=euclidean)
# distance_024_3_hr, path_024_3_hr = fastdtw(x = Y_024_1415_hr, y = Y_024_1819_hr, dist=euclidean)
# distance_024_4_hr, path_024_4_hr = fastdtw(x = Y_024_1415_hr, y = Y_024_1920_hr, dist=euclidean)

# distance_024_0_eda, path_024_0_eda = fastdtw(x = Y_024_1415_eda, y = Y_024_1516_eda, dist=euclidean)
# distance_024_1_eda, path_024_1_eda = fastdtw(x = Y_024_1415_eda, y = Y_024_1617_eda, dist=euclidean)
# distance_024_2_eda, path_024_2_eda = fastdtw(x = Y_024_1415_eda, y = Y_024_1718_eda, dist=euclidean)
# distance_024_3_eda, path_024_3_eda = fastdtw(x = Y_024_1415_eda, y = Y_024_1819_eda, dist=euclidean)
# distance_024_4_eda, path_024_4_eda = fastdtw(x = Y_024_1415_eda, y = Y_024_1920_eda, dist=euclidean)

# distance_024_0_temp, path_024_0_temp = fastdtw(x = Y_024_1415_temp, y = Y_024_1516_temp, dist=euclidean)
# distance_024_1_temp, path_024_1_temp = fastdtw(x = Y_024_1415_temp, y = Y_024_1617_temp, dist=euclidean)
# distance_024_2_temp, path_024_2_temp = fastdtw(x = Y_024_1415_temp, y = Y_024_1718_temp, dist=euclidean)
# distance_024_3_temp, path_024_3_temp = fastdtw(x = Y_024_1415_temp, y = Y_024_1819_temp, dist=euclidean)
# distance_024_4_temp, path_024_4_temp = fastdtw(x = Y_024_1415_temp, y = Y_024_1920_temp, dist=euclidean)

# distance_024_temp = [distance_024_0_temp,distance_024_1_temp,distance_024_2_temp,distance_024_3_temp,distance_024_4_temp]
# distance_024_eda = [distance_024_0_eda,distance_024_1_eda,distance_024_2_eda,distance_024_3_eda,distance_024_4_eda]
# distance_024_hr = [distance_024_0_hr,distance_024_1_hr,distance_024_2_hr,distance_024_3_hr,distance_024_4_hr]
# distance_024_acc_mag = [distance_024_0_acc_mag,distance_024_1_acc_mag,distance_024_2_acc_mag,distance_024_3_acc_mag,distance_024_4_acc_mag]

# plt.figure(figsize=(20,5))

# plt.subplot(141)
# plt.plot(distance_024_temp)
# plt.title('Temp')
# plt.subplot(142)
# plt.title('eda')
# plt.plot(distance_024_eda)
# plt.subplot(143)
# plt.title('hr')
# plt.plot(distance_024_hr)
# plt.subplot(144)
# plt.title('acc_mag')
# plt.plot(distance_024_acc_mag)


# # In[67]:

# plt.figure(figsize=(20,5))
# plt.subplot(141)
# plt.plot(distance_024_temp)
# plt.ylabel('024_distance')
# plt.title('Temp')
# plt.ylim(0,15000)
# plt.subplot(142)
# plt.title('eda')
# plt.plot(distance_024_eda)
# plt.ylim(0,30000)
# plt.subplot(143)
# plt.title('hr')
# plt.plot(distance_024_hr)
# plt.ylim(0,50000)
# plt.subplot(144)
# plt.title('acc_mag')
# plt.ylim(0,10000)
# plt.plot(distance_024_acc_mag)