import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sl import *


def main():
    #result_dir = 'result_mnist.pkl'
    result_dir = 'result_circle.pkl'
    _,_,dataSizes,loss_oracle,error_rate,model_selected,timing,loss_ratio,mode = load(result_dir)
    #dataSizes,loss_oracle,model_selected,timing,loss_ratio,mode = load(result_dir)
    #print(error_rate)
    #print(model_selected)
    showAveLossRatio(loss_ratio,mode)
    viewLossRatioResult(dataSizes,loss_ratio,mode,'loss ratio')
    result_dir = 'timing.pkl'
    dataSizes,loss_oracle,model_selected,timing,loss_ratio,mode = load(result_dir)
    viewTimingResult(dataSizes,timing,mode,'elapsed time')

    
def viewLossRatioResult(dataSizes,result,mode,name):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    plt.figure(figsize=(10, 2), dpi=300, facecolor='w', edgecolor='k')
    linestyles = ['--', '--', '-.', ':']
    labels = ['GTIC','Holdout','10 fold','LOO']
    for i in range(len(mode)):
        plt.plot(dataSizes,result[mode[i]], color=colors[i], label=labels[i], linewidth=1, linestyle=linestyles[i])
    plt.xlim((dataSizes[0],dataSizes[-1]))
    plt.xlabel('Data Size', fontsize=14, color='black')
    plt.ylabel('Loss ratio', fontsize=14, color='black')
    plt.title('Loss ratio of the selected model to the optimal model (oracle)')
    plt.legend(loc='upper right', prop={'size':7})
    plt.grid()
    plt.savefig('./result/lossratio.png', bbox_inches='tight')
    plt.show()

def viewTimingResult(dataSizes,result,mode,name):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    plt.figure(figsize=(10, 2), dpi=300, facecolor='w', edgecolor='k')
    linestyles = ['--', '--', '-.', ':']
    labels = ['GTIC','Holdout','10 fold','LOO']
    mode = ['tic','holdout','10fold']
    for i in range(len(mode)):
        result[mode[i]] = np.cumsum(result[mode[i]])
    for i in range(len(mode)):
        plt.plot(dataSizes,result[mode[i]], color=colors[i], label=labels[i], linewidth=1, linestyle=linestyles[i])
    plt.xlim((dataSizes[0],dataSizes[-1]))
    plt.xlabel('Data Size', fontsize=14, color='black')
    plt.ylabel('Elapsed time (in seconds)', fontsize=14, color='black')
    plt.title('Computational complexity')
    plt.legend(loc='upper right', prop={'size':7})
    plt.grid()
    plt.savefig('./result/timing.png', bbox_inches='tight')
    plt.show()
    
def showAveLossRatio(loss_ratio,mode):
    print(mode)
    for i in range(len(mode)):
        print(mode[i])
        print(np.mean(loss_ratio[mode[i]]))   

if __name__ == "__main__":
    main() 