import torch
import copy
from torch import nn
from torch.autograd import Variable
from data import *
from util import *
from model import *
from modelselect import *


ifcuda = True
learning_rate = 1e-1
init_epochs = 0 
num_epochs = 30
dataSize = 10000
batch_size = 128
input_feature = list(range(5,21))
output_feature = 2
input_datatype = torch.FloatTensor
target_datatype = torch.LongTensor

def main():
    randomGen = np.random.RandomState(seed)
    X, y = fetch_data_logistic(dataSize,randomGen)
    K = 3
    print('Start {} fold validation with Early Stopping'.format(K))
    s = time.time()
    Experiment_LR(X,y,K,)
    e = time.time()
    print('Total Elapsed Time: {}'.format(e-s))
    
def Experiment_LR(X,y,K):
    X_train_all, X_val_all, X_test_all, y_train, y_val, y_test = split_data(X,y,K)
    best_model_val_loss = np.zeros((K,len(input_feature)))
    best_model_epoch_id = np.zeros((K,len(input_feature)),dtype=np.int16)
    for k in range(K):
        for i in range(len(input_feature)):
            X_train, X_val= X_train_all[k][:,:input_feature[i]], X_val_all[k][:,:input_feature[i]]
            print(X_train.shape)
            get_data_stats(X_train,None)
            train_loader = get_data_loader(X_train,y_train[k],input_datatype,target_datatype,batch_size)
            val_loader = get_data_loader(X_val,y_val[k],input_datatype,target_datatype,batch_size)
            if ifcuda:
                model = LogisticRegression(input_feature[i],output_feature).cuda()
            else:
                model = LogisticRegression(input_feature[i],output_feature)
            TAG = 'CrossValidation_{}_{}'.format(k,input_feature[i])
            train_loss_iter,train_loss_epoch = trainer(copy.deepcopy(model),train_loader,TAG)
            val_loss_iter,val_loss_epoch = validater(copy.deepcopy(model),val_loader,TAG)
            best_model_val_loss[k,i] = np.amin(val_loss_epoch)
            best_model_epoch_id[k,i] = np.argmin(val_loss_epoch)
            print(best_model_epoch_id[k,i])
            #showLoss(num_epochs,['train','val'],TAG)
    ## Finalize
    print(best_model_val_loss)
    best_fold_val_loss = np.mean(best_model_val_loss,axis=0)
    best_model_id = np.argmin(best_fold_val_loss)
    print(input_feature[best_model_id])
    sample_k = 0
    X_final_all = np.concatenate((X_train_all[sample_k],X_val_all[sample_k]),axis=0)
    y_final = np.concatenate((y_train[sample_k],y_val[sample_k]),axis=0)
    X_final_train_all, X_final_val_all, y_final_train, y_final_val = split_data_p(X_final_all,y_final,0.75)
    X_final_train, X_final_val = X_final_train_all[:,:input_feature[best_model_id]], X_final_val_all[:,:input_feature[best_model_id]]
    print(X_final_train_all.shape)
    print(X_final_train.shape)
    get_data_stats(X_final_train,None)
    final_train_loader = get_data_loader(X_final_train,y_final_train,input_datatype,target_datatype,batch_size)
    final_val_loader = get_data_loader(X_final_val,y_final_val,input_datatype,target_datatype,batch_size)
    if ifcuda:
        model = LogisticRegression(input_feature[best_model_id],output_feature).cuda()
    else:
        model = LogisticRegression(input_feature[best_model_id],output_feature) 
    TAG = 'CrossValidation_final_{}'.format(input_feature[best_model_id])
    final_train_loss_iter,final_train_loss_epoch = trainer(copy.deepcopy(model),final_train_loader,TAG)
    final_val_loss_iter,final_val_loss_epoch = validater(copy.deepcopy(model),final_val_loader,TAG)
    final_best_model_val_loss = np.amin(final_val_loss_epoch)
    final_best_model_epoch_id = np.argmin(final_val_loss_epoch)
    X_test = X_test_all[:,:input_feature[best_model_id]]
    test_loader = get_data_loader(X_test,y_test,input_datatype,target_datatype,batch_size)
    test_loss_iter, best_model_test_loss = tester(copy.deepcopy(model),final_best_model_epoch_id,test_loader,TAG)
    print(best_model_test_loss)

    
def trainer(model,train_loader,TAG=""):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Training ...")
    dataloader = train_loader
    num_iter = len(dataloader)
    train_loss_iter = np.zeros(num_iter*num_epochs)
    train_loss_epoch = np.zeros(num_epochs)
    timing_epoch = np.zeros(num_epochs)
    i = 0
    j = 0
    for epoch in range(init_epochs,(init_epochs+num_epochs)):
        s = time.time()
        for input,target in dataloader:
            input,_ = normalize(input,None)
            input = to_var(input,ifcuda)
            target = to_var(target,ifcuda).squeeze()
            # ===================forward=====================
            output = model(input).squeeze()
            loss = criterion(output, target)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_iter[i] = loss.data[0]           
            i = i + 1
        e = time.time()
        # ===================log========================
        train_loss_epoch[j] = np.mean(train_loss_iter[(epoch-init_epochs)*num_iter:(epoch-init_epochs+1)*num_iter])
        timing_epoch[j] = e-s
        print("Elapsed Time for one epoch: %.3f" % timing_epoch[j])
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, init_epochs+num_epochs, train_loss_epoch[j]))
        save_model(model, './model/model_{}_{}.pth'.format(epoch,TAG))
        save([train_loss_iter,train_loss_epoch],'./output/train/loss_{}.pkl'.format(TAG))
        save(timing_epoch, './output/train/timing_{}.pkl'.format(TAG))
        j = j + 1
    return train_loss_iter,train_loss_epoch

def validater(model,val_loader,TAG=""):
    criterion = nn.CrossEntropyLoss()
    print("Validating ...")
    dataloader = val_loader
    num_iter = len(dataloader)
    val_loss_iter = np.zeros(num_iter*num_epochs)
    val_loss_epoch = np.zeros(num_epochs)
    i = 0
    j = 0
    for epoch in range(init_epochs,(init_epochs+num_epochs)):
        model = load_model(model,ifcuda,'./model/model_{}_{}.pth'.format(epoch,TAG))
        model = model.eval()
        s = time.time()
        for input,target in dataloader:
            input,_ = normalize(input,None)
            input = to_var(input,ifcuda)
            target = to_var(target,ifcuda).squeeze()
            # ===================forward=====================
            output = model(input).squeeze()
            loss = criterion(output, target)
            val_loss_iter[i] = loss.data[0]           
            i = i + 1
        e = time.time()
        # ===================log========================
        val_loss_epoch[j] = np.mean(val_loss_iter[(epoch-init_epochs)*num_iter:(epoch-init_epochs+1)*num_iter])
        print("Elapsed Time for one epoch: %.3f" % (e-s))
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, init_epochs+num_epochs, val_loss_epoch[j]))
        save([val_loss_iter,val_loss_epoch],'./output/val/loss_{}.pkl'.format(TAG))
        j = j + 1
    return val_loss_iter,val_loss_epoch

def tester(model,epoch_id,test_loader,TAG=""):
    criterion = nn.CrossEntropyLoss()
    print("Testing ...")
    dataloader = test_loader
    num_iter = len(dataloader)
    test_loss_iter = np.zeros(num_iter)
    test_loss = 0
    i = 0
    print('./model/model_{}_{}.pth'.format(epoch_id,TAG))
    model = load_model(model,ifcuda,'./model/model_{}_{}.pth'.format(epoch_id,TAG))
    model = model.eval()
    s = time.time()
    for input,target in dataloader:
        input,_ = normalize(input,None)
        input = to_var(input,ifcuda)
        target = to_var(target,ifcuda).squeeze()
        # ===================forward=====================
        output = model(input).squeeze()
        loss = criterion(output, target)
        test_loss_iter[i] = loss.data[0]           
        i = i + 1
    e = time.time()
    # ===================log========================
    test_loss = np.mean(test_loss_iter)
    print("Elapsed Time for one epoch: %.3f" % (e-s))
    save([test_loss_iter,test_loss],'./output/test/loss_{}.pkl'.format(TAG))
    return test_loss_iter,test_loss
    
if __name__ == "__main__":
    Experiment_LR()