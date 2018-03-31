import torch
import copy
from torch import nn
from torch.autograd import Variable
from data import *
from util import *
from model import *
from modelselect import *


ifcuda = True
learning_rate = 1*1e-3
init_epochs = 0 
max_num_epochs = 30
dataSize = 500
batch_size = 128
min_delta = 5*1e-4
patience = 5
input_datatype = torch.FloatTensor
target_datatype = torch.LongTensor
num_repetition = 1
def main():   
    X, y = fetch_data_mld()
    valid_target = np.array([2,3])
    X, y = filter_data(X,y,valid_target)
    input_features = gen_input_features_LogisticRegression((28,28))
    out_features = [2]*len(input_features)     
    
    # X, y = fetch_data_logistic(dataSize,randomGen)
    # out_features = [2]*len(input_features)
    
    runExperiment(X,y,input_features,out_features,num_repetition=num_repetition)


            
def runExperiment(X,y,input_features,out_features,num_repetition=num_repetition):
    seeds = list(range(num_repetition))
    selected_model_id = np.zeros(len(seeds))
    selected_model_test_loss = np.zeros(len(seeds))
    best_model_test_loss = np.zeros(len(seeds))
    selected_model_test_acc = np.zeros(len(seeds))
    best_model_test_acc = np.zeros(len(seeds))
    efficiency = np.zeros(len(seeds))
    for i in range(len(seeds)):
        print('Start Experiment {} of Regulariztion with Early Stopping'.format(i))  
        randomGen = np.random.RandomState(seeds[i])
        data = gen_Regularization_data_LogisticRegression(X,y,input_features,randomGen)    
        models = gen_models_LogisticRegression(input_features,out_features,ifcuda)
        criterion = nn.CrossEntropyLoss(size_average=False)        
        s = time.time()
        selected_model_id[i],selected_model_test_loss[i],best_model_test_loss[i],selected_model_test_acc[i],best_model_test_acc[i],efficiency[i] = Experiment(i,data,models,criterion)
        e = time.time()
        print('Total Elapsed Time for Experiment {}: {}'.format(i,e-s))        
    save([selected_model_id,selected_model_test_loss,best_model_test_loss,selected_model_test_acc,best_model_test_acc,efficiency],'./output/Experiment_all.pkl')
    
def Experiment(id,data,models,criterion):
    X_train, X_test, y_train, y_test = data
    validated_model_epoch_id = np.zeros(len(models)),dtype=np.int16)
    validated_model_loss = np.zeros(len(models))
    for i in range(len(models)):
            print(X_train[i].shape)
            get_data_stats(X_train[i],None)
            train_loader = get_data_loader(X_train[i],y_train,input_datatype,target_datatype,batch_size)
            TAG = 'Regularization_final_{}'.format(i)
            train_loss_epoch,train_acc_epoch,val_loss_epoch = trainer(train_loader,copy.deepcopy(models[i]),criterion,TAG)
            validated_model_epoch_id[i] = EarlyStopping(val_loss_epoch,min_delta,patience)
            validated_model_loss[i] = val_loss_epoch[validated_model_epoch_id[i]]
            print(validated_model_epoch_id[i])
            # showLoss(max_num_epochs,['train','val'],TAG)

    ## Finalize
    print('Finalizing...')
    selected_model_id = np.argmin(validated_model_loss)
    print(selected_model_id)
    print(X_train[selected_model_id].shape)

    final_validated_model_epoch_id = validated_model_epoch_id
    final_validated_model_loss = validated_model_loss  
    final_model_test_loss = np.zeros(len(models))  
    final_model_test_acc = np.zeros(len(models))     
    for i in range(len(models)):
        test_loader = get_data_loader(X_test[i],y_test,input_datatype,target_datatype,batch_size)
        final_model_test_loss[i], final_model_test_acc[i] = tester(test_loader,copy.deepcopy(models[i]),final_validated_model_epoch_id[i],criterion,TAG)  
     
    final_selected_model_test_loss = final_model_test_loss[selected_model_id]
    final_best_model_test_loss = np.min(final_model_test_loss)
    final_selected_model_test_acc = final_model_test_acc[selected_model_id]
    final_best_model_test_acc = np.max(final_model_test_acc)
    efficiency = final_best_model_test_loss/final_selected_model_test_loss
    print('final validated model epoch:')
    print(final_validated_model_epoch_id)    
    print('best model id: {}\nselected model loss: {}\nbest model loss: {}\nselected model acc: {}\nbest model acc: {}\nefficiency: {}'
        .format(selected_model_id,final_selected_model_test_loss,final_best_model_test_loss,final_selected_model_test_acc,final_best_model_test_acc,efficiency))
    save([validated_model_loss,selected_model_id,final_validated_model_epoch_id,final_model_test_loss,final_model_test_acc,efficiency],'./output/Experiment_{}.pkl'.format(id))
    return selected_model_id,final_selected_model_test_loss,final_best_model_test_loss,final_selected_model_test_acc,final_best_model_test_acc,efficiency
    

    
def trainer(train_loader,model,criterion,TAG=""):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    print("Training ...")
    dataloader = train_loader
    num_iter = len(dataloader)
    train_loss_iter = np.zeros(num_iter*max_num_epochs)
    train_loss_epoch = np.zeros(max_num_epochs)
    train_acc_iter = np.zeros(num_iter*max_num_epochs)
    train_acc_epoch = np.zeros(max_num_epochs)
    val_loss_epoch = np.zeros(max_num_epochs)
    i = 0
    j = 0
    for epoch in range(init_epochs,(init_epochs+max_num_epochs)):
        s = time.time()
        size_tracker = 0
        loss_tracker = 0
        count_tracker = 0
        for input,target in dataloader:
            batch_dataSize = input.shape[0]
            size_tracker += batch_dataSize
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
            loss_tracker += loss.data[0]  
            count_tracker += get_correct_cnt(output,target) 
            train_loss_iter[i] = loss_tracker/size_tracker
            train_acc_iter[i] = count_tracker/size_tracker   
            i = i + 1
        e = time.time()
        # ===================log========================
        train_loss_epoch[j] = train_loss_iter[i-1]
        train_acc_epoch[j] = train_acc_iter[i-1]
        val_loss_epoch = get_AIC(size_tracker,model,train_loss_epoch)
        print("Elapsed Time for one epoch: %.3f" % (e-s))
        print('epoch [{}/{}], loss:{:.4f}, acc:{:.4f}'
            .format(epoch+1, init_epochs+max_num_epochs, train_loss_epoch[j], train_acc_epoch[j]))
        save_model(model, './model/model_{}_{}.pth'.format(epoch,TAG))
        save([train_loss_iter,train_loss_epoch],'./output/train/loss_{}.pkl'.format(TAG))
        save([train_acc_iter,train_acc_epoch],'./output/train/acc_{}.pkl'.format(TAG))
        j = j + 1
    return train_loss_epoch,train_acc_epoch

def tester(test_loader,model,epoch_id,criterion,TAG=""):
    print("Testing ...")
    dataloader = test_loader
    num_iter = len(dataloader)
    test_loss_iter = np.zeros(num_iter)
    test_loss = 0
    test_acc_iter = np.zeros(num_iter)
    test_acc = 0
    i = 0
    print('./model/model_{}_{}.pth'.format(epoch_id,TAG))
    model = load_model(model,ifcuda,'./model/model_{}_{}.pth'.format(epoch_id,TAG))
    model = model.eval()
    s = time.time()
    size_tracker = 0
    loss_tracker = 0
    count_tracker = 0
    for input,target in dataloader:
        batch_dataSize = input.shape[0]
        size_tracker += batch_dataSize
        input,_ = normalize(input,None)
        input = to_var(input,ifcuda)
        target = to_var(target,ifcuda).squeeze()
        # ===================forward=====================
        output = model(input).squeeze()
        loss = criterion(output, target)
        loss_tracker += loss.data[0]  
        count_tracker += get_correct_cnt(output,target) 
        test_loss_iter[i] = loss_tracker/size_tracker 
        test_acc_iter[i] = count_tracker/size_tracker          
        i = i + 1
    e = time.time()
    # ===================log========================
    test_loss = test_loss_iter[i-1]
    test_acc = test_acc_iter[i-1]
    print("Elapsed Time for one epoch: %.3f" % (e-s))
    print('loss:{:.4f}, acc:{:.4f}'
        .format(test_loss, test_acc))
    save([test_loss_iter,test_loss],'./output/test/loss_{}.pkl'.format(TAG))
    save([test_acc_iter,test_acc],'./output/test/acc_{}.pkl'.format(TAG))
    return test_loss,test_acc
    
if __name__ == "__main__":
    main()