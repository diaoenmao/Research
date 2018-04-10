import torch
import copy
from torch import nn
from torch.autograd import Variable
from data import *
from util import *
from model import *
from modelselect import *
from runner import *


class stochastic_runner(runner):
    
    def __init__(self, id, data, modelwrappers, criterion, batch_size = 128, ifcuda = True, verbose = True, ifsave = True):
        super().__init__(id, data, modelwrappers, criterion, ifcuda, verbose, ifsave)
        self.num_models = len(self.modelwrappers)
        self.batch_size = batch_size
        
        self.mode = 'CrossValidation_1'
        self.modename = 'CrossValidation'
        self.input_datatype = torch.FloatTensor
        self.target_datatype = torch.LongTensor
        self.max_num_epochs = 30
        self.min_delta = 5*1e-4
        self.patience = 5
        
    def set_mode(self, mode = 'CrossValidation'):
        self.mode = mode
    
    def set_datatype(self, input_datatype = torch.FloatTensor, target_datatype = torch.LongTensor):
        self.input_datatype = input_datatype
        self.target_datatype = target_datatype
        
    def set_early_stopping(self, max_num_epochs = 30, min_delta = 5*1e-4, patience = 5):
        self.max_num_epochs = max_num_epochs
        self.min_delta = min_delta
        self.patience = patience
        
    def train(self,ifshow=False):
        X_train, X_val, X_final, _, y_train, y_val, y_final, _ = self.data
        self.K = len(X_train[0])
        TAG = self.mode + '_' + str(self.K)
        self.validated_model_loss = np.zeros((self.num_models,self.K,self.max_num_epochs))        
        self.validated_model_epoch_id = np.zeros(self.num_models,dtype=np.int16)
        self.modelselect_loss = np.zeros(self.num_models)
        for i in range(self.num_models):
            for k in range(self.K):
                print(X_train[i][k].shape)
                cur_TAG = TAG + '_{}_{}'.format(i,k)
                get_data_stats(X_train[i][k],TAG=cur_TAG)
                train_loader = get_data_loader(X_train[i][k],y_train[k],self.input_datatype,self.target_datatype,self.batch_size)
                val_loader = get_data_loader(X_val[i][k],y_val[k],self.input_datatype,self.target_datatype,self.batch_size)
                self.trainer(train_loader,copy.deepcopy(self.modelwrappers[i]),self.criterion,cur_TAG)
                self.validated_model_loss[i,k,:],_ = self.validater(val_loader,copy.deepcopy(self.modelwrappers[i]),self.criterion,cur_TAG)
                if(ifshow):
                    showLoss(self.max_num_epochs,['train','val'],cur_TAG)
        mean_validated_model_loss = np.mean(self.validated_model_loss,axis=1)
        for i in range(self.num_models):        
            self.validated_model_epoch_id[i] = early_stopping(mean_validated_model_loss[i,:],self.min_delta,self.patience)
        print(self.validated_model_epoch_id)
        if(self.mode == "CrossValidation"):                        
            for i in range(self.num_models):        
                self.modelselect_loss[i] = mean_validated_model_loss[i,self.validated_model_epoch_id[i]]       
            self.selected_model_id = np.argmin(self.modelselect_loss)
                     
            print('Finalizing...')
            for i in range(self.num_models):
                print(X_final[i].shape)            
                cur_TAG = TAG + '_{}_final'.format(i)
                get_data_stats(X_final[i],TAG=cur_TAG)           
                train_loader = get_data_loader(X_final[i],y_final,self.input_datatype,self.target_datatype,self.batch_size)
                self.trainer(train_loader,copy.deepcopy(self.modelwrappers[i]),self.criterion,cur_TAG)
                
        elif(self.mode in self.regulazition_supported):                                        
            print('Finalizing...')
            self.dataSizes = np.zeros(self.num_models)
            for i in range(self.num_models):  
                print(X_final[i].shape)             
                cur_TAG = TAG + '_{}_final'.format(i)
                self.dataSizes[i] = X_final[i].shape[0]
                get_data_stats(X_final[i],TAG=cur_TAG)           
                train_loader = get_data_loader(X_final[i],y_final,self.input_datatype,self.target_datatype,self.batch_size)
                self.trainer(train_loader,copy.deepcopy(self.modelwrappers[i]),self.criterion,cur_TAG)
                self.modelselect_loss[i],_ = self.tester(train_loader,copy.deepcopy(self.modelwrappers[i]),self.validated_model_epoch_id[i],self.criterion,cur_TAG) 
            
            models,_ = unpack_modelwrappers(self.modelwrappers)
            if(self.mode=='Base'):
                self.modelselect_loss = self.modelselect_loss
            if(self.mode=='AIC'):
                self.modelselect_loss = self.modelselect_loss + get_AIC(self.dataSizes,models,self.modelselect_loss)
            elif(self.mode=='BIC'):
                self.modelselect_loss = self.modelselect_loss + get_BIC(self.dataSizes,models,self.modelselect_loss)
            elif(self.mode=='BC'):
                self.modelselect_loss = self.modelselect_loss + get_BC(self.dataSizes,models,self.modelselect_loss)
            elif(self.mode=='TIC'):
                self.modelselect_loss = self.modelselect_loss + get_TIC(self.dataSizes,models,self.modelselect_loss)
                
            self.selected_model_id = np.argmin(self.modelselect_loss)  
            
    def test(self):
        TAG = self.mode + '_' + str(self.K)
        _, _, _, X_test, _, _, _, y_test = self.data
        self.final_model_test_loss = np.zeros(self.num_models)  
        self.final_model_test_acc = np.zeros(self.num_models) 
        for i in range(self.num_models):
            cur_TAG = TAG + '_{}_final'.format(i)
            test_loader = get_data_loader(X_test[i],y_test,self.input_datatype,self.target_datatype,self.batch_size)
            self.final_model_test_loss[i], self.final_model_test_acc[i] = self.tester(test_loader,copy.deepcopy(self.modelwrappers[i]),self.validated_model_epoch_id[i],self.criterion,cur_TAG)  
        self.best_model_id = np.argmin(self.final_model_test_loss)
        self.final_selected_model_test_loss = self.final_model_test_loss[self.selected_model_id]
        self.final_best_model_test_loss = np.min(self.final_model_test_loss)
        self.final_selected_model_test_acc = self.final_model_test_acc[self.selected_model_id]
        self.final_best_model_test_acc = np.max(self.final_model_test_acc)
        self.efficiency = self.final_best_model_test_loss/self.final_selected_model_test_loss   
        print('selected model id: {}\nbest model id: {}\nselected model loss: {}\nbest model loss: {}\nselected model acc: {}\nbest model acc: {}\nefficiency: {}'
            .format(self.selected_model_id,self.best_model_id,self.final_selected_model_test_loss,self.final_best_model_test_loss,self.final_selected_model_test_acc,self.final_best_model_test_acc,self.efficiency))
        if(self.ifsave):
            save([self.validated_model_loss,self.selected_model_id,self.validated_model_epoch_id,self.final_model_test_loss,self.final_model_test_acc,self.efficiency],'./output/Experiment_{}_{}.pkl'.format(TAG,self.id))
        return self.selected_model_id,self.best_model_id,self.final_selected_model_test_loss,self.final_best_model_test_loss,self.final_selected_model_test_acc,self.final_best_model_test_acc,self.efficiency
        
    def get_output(self):
        return self.selected_model_id,self.best_model_id,self.final_selected_model_test_loss,self.final_best_model_test_loss,self.final_selected_model_test_acc,self.final_best_model_test_acc,self.efficiency
        
    def trainer(self,train_loader,modelwrapper,criterion,TAG=""):
        print("Training ...")
        model = modelwrapper.model
        optimizer = modelwrapper.optimizer
        dataloader = train_loader
        num_iter = len(dataloader)
        train_loss_iter = np.zeros(num_iter*self.max_num_epochs)
        train_loss_epoch = np.zeros(self.max_num_epochs)
        train_acc_iter = np.zeros(num_iter*self.max_num_epochs)
        train_acc_epoch = np.zeros(self.max_num_epochs)
        i = 0
        j = 0
        for epoch in range(self.max_num_epochs):
            s = time.time()
            size_tracker = 0
            loss_tracker = 0
            count_tracker = 0
            for input,target in dataloader:
                batch_dataSize = input.shape[0]
                size_tracker += batch_dataSize
                input,_ = normalize(input,TAG=TAG)
                input = to_var(input,self.ifcuda)
                target = to_var(target,self.ifcuda)
                # ===================forward=====================
                output = model(input)
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
            if(self.verbose):
                print("Elapsed Time for one epoch: %.3f" % (e-s))
                print('epoch [{}/{}], loss:{:.4f}, acc:{:.4f}'
                    .format(epoch+1, self.max_num_epochs, train_loss_epoch[j], train_acc_epoch[j]))
            if(self.ifsave):
                save_model(model, './model/model_{}_{}.pth'.format(epoch,TAG))
                save([train_loss_iter,train_loss_epoch],'./output/train/loss_{}.pkl'.format(TAG))
                save([train_acc_iter,train_acc_epoch],'./output/train/acc_{}.pkl'.format(TAG))
            j = j + 1
        return train_loss_epoch,train_acc_epoch

    def validater(self,val_loader,modelwrapper,criterion,TAG=""):
        print("Validating ...")
        model = modelwrapper.model
        dataloader = val_loader
        num_iter = len(dataloader)
        val_loss_iter = np.zeros(num_iter*self.max_num_epochs)
        val_loss_epoch = np.zeros(self.max_num_epochs)
        val_acc_iter = np.zeros(num_iter*self.max_num_epochs)
        val_acc_epoch = np.zeros(self.max_num_epochs)
        i = 0
        j = 0
        for epoch in range(self.max_num_epochs):
            model = load_model(model,self.ifcuda,'./model/model_{}_{}.pth'.format(epoch,TAG))
            model = model.eval()
            s = time.time()
            size_tracker = 0
            loss_tracker = 0
            count_tracker = 0
            for input,target in dataloader:
                batch_dataSize = input.shape[0]
                size_tracker += batch_dataSize
                input,_ = normalize(input,TAG=TAG)
                input = to_var(input,self.ifcuda)
                target = to_var(target,self.ifcuda)
                # ===================forward=====================
                output = model(input)
                loss = criterion(output, target)
                loss_tracker += loss.data[0]  
                count_tracker += get_correct_cnt(output,target) 
                val_loss_iter[i] = loss_tracker/size_tracker        
                val_acc_iter[i] = count_tracker/size_tracker          
                i = i + 1
            e = time.time()
            # ===================log========================
            val_loss_epoch[j] = val_loss_iter[i-1]
            val_acc_epoch[j] = val_acc_iter[i-1]
            if(self.verbose):
                print("Elapsed Time for one epoch: %.3f" % (e-s))
                print('epoch [{}/{}], loss:{:.4f}, acc:{:.4f}'
                    .format(epoch+1, self.max_num_epochs, val_loss_epoch[j], val_acc_epoch[j]))
            if(self.ifsave):
                save([val_loss_iter,val_loss_epoch],'./output/val/loss_{}.pkl'.format(TAG))
                save([val_acc_iter,val_acc_epoch],'./output/val/acc_{}.pkl'.format(TAG))
            j = j + 1
        return val_loss_epoch,val_acc_epoch
        
    def tester(self,test_loader,modelwrapper,epoch,criterion,TAG=""):
        print("Testing ...")
        model = modelwrapper.model
        dataloader = test_loader
        num_iter = len(dataloader)
        test_loss_iter = np.zeros(num_iter)
        test_loss = 0
        test_acc_iter = np.zeros(num_iter)
        test_acc = 0
        i = 0
        print('./model/model_{}_{}.pth'.format(epoch,TAG))
        model = load_model(model,self.ifcuda,'./model/model_{}_{}.pth'.format(epoch,TAG))
        model = model.eval()
        s = time.time()
        size_tracker = 0
        loss_tracker = 0
        count_tracker = 0
        for input,target in dataloader:
            batch_dataSize = input.shape[0]
            size_tracker += batch_dataSize
            input,_ = normalize(input,TAG=TAG)
            input = to_var(input,self.ifcuda)
            target = to_var(target,self.ifcuda)
            # ===================forward=====================
            output = model(input)
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
    
    