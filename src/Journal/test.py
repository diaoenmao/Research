from util import *
from data import *


# X, y = fetch_data_mld()
# valid_target = np.array([5])
# X, y = filter_data(X,y,valid_target)
# first_image = X[0,:]
# first_image = np.array(first_image, dtype='float')
# pixels = first_image.reshape((28, 28))
# plt.ion()
# plt.figure()
# plt.imshow(pixels, cmap='gray')
# plt.show()



# first_image = X[0,:]  
# first_image = np.array(first_image, dtype='float')      
# dims = (28,28)
# input_features = gen_input_features_LR(dims)
# print(len(input_features))
# for i in range(len(input_features)):
    # cur_img = first_image[input_features[i]].reshape((np.int(np.sqrt(input_features[i].shape[0])),np.int(np.sqrt(input_features[i].shape[0]))))
    # plt.figure()
    # plt.imshow(cur_img, cmap='gray')
    # plt.show()
    
# plt.pause(10000)


#'Base_1','AIC_1','BIC_1','BC_1','CrossValidation_1','CrossValidation_3','CrossValidation_10','GTIC_1','Lasso_1','Ridge_1','ElasticNet_1','GREG_1'
mode = 'BIC_1'
selected_model_id,best_model_id,selected_model_test_loss,best_model_test_loss,selected_model_test_acc,best_model_test_acc,efficiency,timing = load('./output/Experiment_{}.pkl'.format(mode))
print(efficiency)
print(np.mean(efficiency[efficiency!=0]))
print(np.std(efficiency[efficiency!=0]))