import torch
import config
import numbers
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import *
from utils import dict_to_device
config.init()

def PSNR(output,target,max=1.0):
    MAX = torch.tensor(max).to(target.device)
    criterion = nn.MSELoss().to(target.device)
    MSE = criterion(output,target)
    psnr = (20*torch.log10(MAX)-10*torch.log10(MSE)).item()
    return psnr
    
def BPP(code,input_img):
    nbytes = code.numpy().nbytes
    num_pixel = input_img.numel()/input_img.size(1)
    bpp = 8*nbytes/num_pixel
    return bpp

def flatten_output(output):
    if(output['child'] is None):
        return F.log_softmax(output['this'],1)
    else:
        flat_output = []
        for i in range(len(output['child'])):
            output_i = F.log_softmax(output['this'],1)[:,[i]]
            flat_output.append(output_i+flatten_output(output['child'][i]))
        flat_output = torch.cat(flat_output,1)
    return flat_output
        
def ACC(output,target,topk=1):  
    with torch.no_grad():
        batch_size = target.size(0)
        if(isinstance(topk,list)):
            maxk = min(max(topk),output.size(1))
            pred_maxk = output.topk(maxk, 1, True, True)[1].numpy()
            for k in topk:
                if(k<=maxk):
                    pred_k = pred_maxk[:,:k]
                    correct_k = pred_k.eq(target.view(-1,1).expand_as(pred_k)).float().sum()
                    cur_acc = (correct*(100.0 / batch_size)).item()
                    acc.append(cur_acc)
                else:
                    acc.append(1.0)
        else:
            pred_k = output.topk(topk, 1, True, True)[1]
            correct_k = pred_k.eq(target.view(-1,1).expand_as(pred_k)).float().sum()
            acc = (correct_k*(100.0 / batch_size)).item()
    return acc
    
    
def F1(output,target,topk=1):  
    with torch.no_grad():
        if(isinstance(topk,list)):
            maxk = min(max(topk),output.size(1))
            pred_maxk = output.topk(maxk, 1, True, True)[1].numpy()
            for k in topk:
                if(k<=maxk):
                    pred_k = pred_maxk[:,:k]
                    correct_k = pred_k.eq(target.view(-1,1).expand_as(pred_k)).sum(dim=1).byte()
                    pred = pred_k[:,0]
                    pred[correct_k,] = target[correct_k,]
                    cur_f1 = f1_score(target.numpy(),pred.numpy(),average='macro')
                    f1.append(cur_f1)
                else:
                    f1.append(1.0)
        else:
            pred_k = output.topk(topk, 1, True, True)[1]
            correct_k = pred_k.eq(target.view(-1,1).expand_as(pred_k)).sum(dim=1).byte()
            pred = pred_k[:,0]
            pred[correct_k,] = target[correct_k,]
            f1 = f1_score(target.numpy(),pred.numpy(),average='macro')
    return f1

def Precision(output,target,topk=1):  
    with torch.no_grad():
        if(isinstance(topk,list)):
            maxk = min(max(topk),output.size(1))
            pred_maxk = output.topk(maxk, 1, True, True)[1].numpy()
            for k in topk:
                if(k<=maxk):
                    pred_k = pred_maxk[:,:k]
                    correct_k = pred_k.eq(target.view(-1,1).expand_as(pred_k)).sum(dim=1).byte()
                    pred = pred_k[:,0]
                    pred[correct_k,] = target[correct_k,]
                    cur_precision = precision_score(target.numpy(),pred.numpy(),average='macro')
                    precision.append(cur_precision)
                else:
                    precision.append(1.0)
        else:
            pred_k = output.topk(topk, 1, True, True)[1]
            correct_k = pred_k.eq(target.view(-1,1).expand_as(pred_k)).sum(dim=1).byte()
            pred = pred_k[:,0]
            pred[correct_k,] = target[correct_k,]
            precision = precision_score(target.numpy(),pred.numpy(),average='macro')
    return precision

def Recall(output,target,topk=1):  
    with torch.no_grad():
        if(isinstance(topk,list)):
            maxk = min(max(topk),output.size(1))
            pred_maxk = output.topk(maxk, 1, True, True)[1].numpy()
            for k in topk:
                if(k<=maxk):
                    pred_k = pred_maxk[:,:k]
                    correct_k = pred_k.eq(target.view(-1,1).expand_as(pred_k)).sum(dim=1).byte()
                    pred = pred_k[:,0]
                    pred[correct_k,] = target[correct_k,]
                    cur_recall = recall_score(target.numpy(),pred.numpy(),average='macro')
                    recall.append(cur_recall)
                else:
                    recall.append(1.0)
        else:
            pred_k = output.topk(topk, 1, True, True)[1]
            correct_k = pred_k.eq(target.view(-1,1).expand_as(pred_k)).sum(dim=1).byte()
            pred = pred_k[:,0]
            pred[correct_k,] = target[correct_k,]
            recall = recall_score(target.numpy(),pred.numpy(),average='macro')
    return recall
    
class Meter_Panel(object):

    def __init__(self,meter_names):
        self.meter_names = meter_names
        self.panel = {k: Meter() for k in meter_names}
        self.metric = Metric(meter_names)

    def reset(self):
        for k in self.panel:
            self.panel[k].reset()
        self.metric.reset()
        return
        
    def update(self, new, n=1):
        if(isinstance(new, Meter_Panel)):
            for i in range(len(new.meter_names)):
                if(new.meter_names[i] in self.panel):
                    self.panel[new.meter_names[i]].update(new.panel[new.meter_names[i]])
                else:
                    self.panel[new.meter_names[i]] = new.panel[new.meter_names[i]]
                    self.meter_names += [new.meter_names[i]]
        elif(isinstance(new, dict)):
            for k in new:
                if(k not in self.panel):
                    self.panel[k] = Meter()
                    self.meter_names += [k]
                if(isinstance(n,int)):
                    self.panel[k].update(new[k],n)
                else:
                    self.panel[k].update(new[k],n[k])
        else:
            raise ValueError('Not supported data type for updating meter panel')
        return
        
    def eval(self, input, output, protocol):
        tuning_param = protocol['tuning_param']
        metric_names = protocol['metric_names']
        evaluation = self.metric.eval(input,output,protocol)
        return evaluation
        
    def summary(self,names):
        fmt_str = ''
        if('loss' in names and 'loss' in self.panel):
            fmt_str += '\tLoss: {:.4f}'.format(self.panel['loss'].avg)
        if('psnr' in names and 'psnr' in self.panel):
            fmt_str += '\tPSNR: {:.4f}'.format(self.panel['psnr'].avg)
        if('acc' in names and 'acc' in self.panel):
            fmt_str += '\tACC: {:.4f}'.format(self.panel['acc'].avg)
        if('batch_time' in names and 'batch_time' in self.panel):
            fmt_str += '\tTime: {:.4f}'.format(self.panel['batch_time'].sum)
        return fmt_str
                    
                
class Meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history_val = []
        self.history_avg = [0]
        return
        
    def update(self, new, n=1):
        if(isinstance(new,Meter)):
            self.val = new.val
            self.avg = new.avg
            self.sum = new.sum
            self.count = new.count
            self.history_val.extend(new.history_val)
            self.history_avg.extend(new.history_avg)
        elif(isinstance(new,numbers.Number)):
            self.val = new
            self.sum += new * n
            self.count += n
            self.avg = self.sum / self.count
            self.history_val.append(self.val)
            self.history_avg[-1] = self.avg
        else:
            self.val = new
            self.count += n
            self.history_val.append(self.val)
        return
        
        
class Metric(object):
    
    batch_metric_names = ['psnr','bpp','acc']
    full_metric_names = ['f1','precsion','recall','prc','roc','roc_auc']
    
    def __init__(self, metric_names):
        self.reset(metric_names)
        
    def reset(self, metric_names):
        self.metric_names = metric_names
        self.if_save = not set(self.metric_names).isdisjoint(self.full_metric_names)
        self.score = None
        self.label = None
        return
        
    def eval(self, input, output, protocol):
        tuning_param = protocol['tuning_param']
        metric_names = protocol['metric_names']
        evaluation = {}
        evaluation['loss'] = output['loss'].item()
        if(tuning_param['compression'] > 0):
            if('psnr' in metric_names):
                evaluation['psnr'] = PSNR(output['compression'],input['img'])
            if('bpp' in metric_names):
                evaluation['bpp'] = PSNR(output['code'],input['img'])
        if(tuning_param['classification'] > 0):
            topk=protocol['topk']
            if(self.if_save):
                self.score = torch.cat(self.score,output['classification'].cpu(),0) if self.score is not None else output['classification'].cpu()
                self.label = torch.cat(self.label,input['label'].cpu(),0) if self.label is not None else input['label'].cpu()
            if('acc' in metric_names):
                evaluation['acc'] = ACC(output['classification'],input['label'],topk=topk)
            if('f1' in metric_names):
                evaluation['f1'] = F1(self.score,self.label,topk=topk)
            if('precision' in metric_names):
                evaluation['precision'] = Precision(self.score.self.label,topk=topk)
            if('recall' in metric_names):
                evaluation['recall'] = Recall(self.score,self.label,topk=topk)
            if('prc' in metric_names):
                evaluation['prc'] = PRC(self.score,self.label)
            if('roc' in metric_names):
                evaluation['roc'] = ROC(self.score,self.label)
            if('roc_auc' in metric_names):
                evaluation['roc_auc'] = ROC_AUC(self.score,self.label)
        return evaluation









        