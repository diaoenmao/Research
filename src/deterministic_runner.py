import torch
import copy
from torch import nn
from torch.autograd import Variable
from data import *
from util import *
from model import *
from modelselect import *
from runner import *


class deterministic_runner(runner):
    
    def __init__(self, id, data, modelwrappers, ifcuda = True, verbose = True, ifsave = True):
        super().__init__(id, data, modelwrappers, ifcuda, verbose, ifsave)      
        self.mode = 'CrossValidation'
        self.input_datatype = torch.FloatTensor
        self.target_datatype = torch.LongTensor
        self.max_num_epochs = 5
        
    def set_mode(self, mode = 'CrossValidation'):
        self.mode = mode
    
    def set_datatype(self, input_datatype = torch.FloatTensor, target_datatype = torch.LongTensor):
        self.input_datatype = input_datatype
        self.target_datatype = target_datatype
          
    def set_max_num_epochs(self,max_num_epochs=5):
        self.max_num_epochs = max_num_epochs
        
    def train(self,ifshow=False):
        X_train, X_val, X_final, _, y_train, y_val, y_final, _ = self.data
        self.K = len(X_train[0])
        TAG = self.mode + '_' + str(self.K)
        self.validated_model_loss = np.zeros((self.num_models,self.K))  
        if(self.mode == "CrossValidation"):        
            self.modelselect_loss = np.zeros(self.num_models)
            for i in range(self.num_models):
                for k in range(self.K):
                    print(X_train[i][k].shape)
                    cur_TAG = TAG + '_{}_{}'.format(i,k)
                    get_data_stats(X_train[i][k],TAG=cur_TAG)
                    train_tensorset = get_data_tensorset(X_train[i][k],y_train[k],self.input_datatype,self.target_datatype)
                    val_tensorset = get_data_tensorset(X_val[i][k],y_val[k],self.input_datatype,self.target_datatype)
                    copied_mw = self.modelwrappers[i].copy()
                    validated_mw = self.trainer(train_tensorset,copied_mw,self.mode,cur_TAG)
                    self.validated_model_loss[i,k],_,_ = self.tester(val_tensorset,validated_mw,self.max_num_epochs-1,cur_TAG)
                    if(ifshow):
                        showLoss(self.max_num_epochs,['train','val'],cur_TAG)

            self.modelselect_loss = np.mean(self.validated_model_loss,axis=1)
            self.selected_model_id = np.argmin(self.modelselect_loss)
                     
            print('Finalizing...')
            for i in range(self.num_models):
                print(X_final[i].shape)            
                cur_TAG = TAG + '_{}_final'.format(i)
                get_data_stats(X_final[i],TAG=cur_TAG)           
                train_loader = get_data_tensorset(X_final[i],y_final,self.input_datatype,self.target_datatype)
                copied_mw = self.modelwrappers[i].copy()
                self.trainer(train_loader,copied_mw,self.mode,cur_TAG)

        elif(self.mode in self.regulazition_supported):                                        
            print('Finalizing...')
            self.dataSizes = np.zeros(self.num_models)
            self.modelselect_loss = []
            self.finalized_modelwrappers = []
            for i in range(self.num_models):  
                print(X_final[i].shape)
                cur_TAG = TAG + '_{}_final'.format(i)
                self.dataSizes[i] = X_final[i].shape[0]
                get_data_stats(X_final[i],TAG=cur_TAG)           
                train_tensorset = get_data_tensorset(X_final[i],y_final,self.input_datatype,self.target_datatype)
                copied_mw = self.modelwrappers[i].copy()
                finalized_model = self.trainer(train_tensorset,copied_mw,self.mode,cur_TAG)
                self.finalized_modelwrappers.append(finalized_model)
                _,_,modelselect_loss_batch = self.tester(train_tensorset,self.finalized_modelwrappers[i],self.max_num_epochs-1,cur_TAG) 
                self.modelselect_loss.append(modelselect_loss_batch)
            self.modelselect_loss = regularization(self.dataSizes,self.finalized_modelwrappers,self.modelselect_loss,self.mode)
            print(self.modelselect_loss)
            self.selected_model_id = np.argmin(self.modelselect_loss)
        else:
            print('mode not supported')
            exit()
            
    def test(self):
        TAG = self.mode + '_' + str(self.K)
        _, _, _, X_test, _, _, _, y_test = self.data
        self.final_model_test_loss = np.zeros(self.num_models)  
        self.final_model_test_acc = np.zeros(self.num_models) 
        for i in range(self.num_models):
            cur_TAG = TAG + '_{}_final'.format(i)
            test_tensorset = get_data_tensorset(X_test[i],y_test,self.input_datatype,self.target_datatype)
            copied_mw = self.modelwrappers[i].copy()
            self.final_model_test_loss[i], self.final_model_test_acc[i],_ = self.tester(test_tensorset,copied_mw,self.max_num_epochs-1,cur_TAG)  
        self.best_model_id = np.argmin(self.final_model_test_loss)
        self.final_selected_model_test_loss = self.final_model_test_loss[self.selected_model_id]
        self.final_best_model_test_loss = np.min(self.final_model_test_loss)
        self.final_selected_model_test_acc = self.final_model_test_acc[self.selected_model_id]
        self.final_best_model_test_acc = np.max(self.final_model_test_acc)
        self.efficiency = self.final_best_model_test_loss/self.final_selected_model_test_loss   
        print('selected model id: {}\nbest model id: {}\nselected model loss: {}\nbest model loss: {}\nselected model acc: {}\nbest model acc: {}\nefficiency: {}'
            .format(self.selected_model_id,self.best_model_id,self.final_selected_model_test_loss,self.final_best_model_test_loss,self.final_selected_model_test_acc,self.final_best_model_test_acc,self.efficiency))
        if(self.ifsave):
            save([self.validated_model_loss,self.selected_model_id,self.final_model_test_loss,self.final_model_test_acc,self.efficiency],'./output/Experiment_{}_{}.pkl'.format(TAG,self.id))
        return self.selected_model_id,self.best_model_id,self.final_selected_model_test_loss,self.final_best_model_test_loss,self.final_selected_model_test_acc,self.final_best_model_test_acc,self.efficiency
        
    def get_output(self):
        return self.selected_model_id,self.best_model_id,self.final_selected_model_test_loss,self.final_best_model_test_loss,self.final_selected_model_test_acc,self.final_best_model_test_acc,self.efficiency
        
    def trainer(self,train_tensorset,modelwrapper,mode,TAG=""):
        print("Training ...")
        model = modelwrapper.model
        optimizer = modelwrapper.optimizer
        input,target = train_tensorset.data_tensor,train_tensorset.target_tensor
        input,_ = normalize(input,input_datatype=self.input_datatype,target_datatype=self.target_datatype,TAG=TAG)
        input = to_var(input,self.ifcuda)
        target = to_var(target,self.ifcuda)
        train_loss_iter = []
        train_loss_epoch = np.zeros(self.max_num_epochs)
        train_regularized_loss_iter = []
        train_regularized_loss_epoch = np.zeros(self.max_num_epochs)
        train_acc_iter = []
        train_acc_epoch = np.zeros(self.max_num_epochs)
        j = 0     
        for epoch in range(self.max_num_epochs):
            s = time.time()
            def closure():
                optimizer.zero_grad()
                # ===================forward=====================
                loss,regularized_loss,loss_batch,acc = modelwrapper.loss_acc(input,target)
                train_loss_iter.append(float(loss)) 
                train_regularized_loss_iter.append(float(regularized_loss))  
                train_acc_iter.append(float(acc))  
                # ===================backward====================
                if(self.ifregularize):
                    print('a')
                    print(modelwrapper.parameters())
                    print(loss)
                    print(regularized_loss)
                    regularized_loss.backward()
                    return regularized_loss
                else:
                    loss.backward()
                    return loss
            optimizer.step(closure)       
            e = time.time()
            # ===================log========================
            train_loss_epoch[j] = train_loss_iter[-1]
            train_regularized_loss_epoch[j] = train_regularized_loss_iter[-1]
            train_acc_epoch[j] = train_acc_iter[-1]
            if(self.verbose):
                print("Elapsed Time for one epoch: %.3f" % (e-s))
                print('epoch [{}/{}], loss:{}, regularized loss:{}, acc:{:.4f}'
                    .format(epoch+1, self.max_num_epochs, train_loss_epoch[j], train_regularized_loss_epoch[j], train_acc_epoch[j]))            
            if(self.ifsave):
                save_model(model, './model/model_{}_{}.pth'.format(epoch,TAG))
                save([train_loss_iter,train_loss_epoch,train_regularized_loss_iter,train_regularized_loss_epoch,train_acc_iter,train_acc_epoch],'./output/train/loss_acc_{}.pkl'.format(TAG))
            j = j + 1
        return modelwrapper
        
    def tester(self,test_tensorset,modelwrapper,epoch,TAG=""):
        print("Testing ...")
        model = modelwrapper.model
        input,target = test_tensorset.data_tensor,test_tensorset.target_tensor
        input,_ = normalize(input,input_datatype=self.input_datatype,target_datatype=self.target_datatype,TAG=TAG)
        input = to_var(input,self.ifcuda)
        target = to_var(target,self.ifcuda)
        print('./model/model_{}_{}.pth'.format(epoch,TAG))
        model = load_model(model,self.ifcuda,'./model/model_{}_{}.pth'.format(epoch,TAG))
        model = model.eval()
        s = time.time()
        # ===================forward=====================
        test_loss,test_regularized_loss,test_loss_batch,test_acc = modelwrapper.loss_acc(input,target)
        e = time.time()
        # ===================log========================
        print("Elapsed Time for one epoch: %.3f" % (e-s))
        print('loss:{}, regularized_loss:{}, acc:{:.4f}'.format(float(test_loss), float(test_regularized_loss), float(test_acc)))
        if(self.ifsave):
            save([test_loss, test_regularized_loss, test_acc],'./output/test/loss_{}.pkl'.format(TAG))
        return test_loss,test_acc,test_loss_batch
    
    