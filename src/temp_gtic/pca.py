import numpy as np
from sklearn.decomposition import PCA

from data import *
from train import *
from sl import *

def pca(n_components,dataX):
    pca= PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(dataX)
    dataX_pca = pca.transform(dataX) 
    return dataX_pca
    
    
def main():
    n_components = 16
    pca_dir = './data/mnist_pca.pkl'
    select_num = np.array([1,2])
    dataX,dataY = filt_mnist_data(select_num)
    dataX_pca = pca(n_components,dataX)
    print(dataX_pca.shape)
    print(dataY.shape)
    save([dataX_pca,dataY],pca_dir)
    
if __name__ == "__main__":
    main() 