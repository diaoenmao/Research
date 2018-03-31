import numpy as np
from sklearn import datasets
from sklearn.datasets.mldata import fetch_mldata
from six.moves import urllib
from sl import *
from pca import *
import os.path
import gzip

def generate_mnist_data(dataSize):
    data_dir = './data/mnist.pkl'
    if(os.path.exists(data_dir)):
        dataX, dataY = load(data_dir)
    else:
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, data_dir)
        print('... loading data')
        # Load the dataset
        with gzip.open(data_dir, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)
        dataX = np.concatenate((train_set[0],valid_set[0],test_set[0]),axis=0)
        dataY = np.concatenate((train_set[1],valid_set[1],test_set[1]),axis=0)             
        save([dataX, dataY], data_dir)
    perm = np.random.permutation(dataX.shape[0])
    dataX = dataX[perm]
    dataY = dataY[perm]
    return dataX[:dataSize,], dataY[:dataSize]

def generate_mnist_pca_data(dataSize,n_components,select_num):
    data_dir = './data/mnist_pca.pkl'
    dataX,dataY = filt_mnist_data(select_num)
    dataX_pca = pca(n_components,dataX)
    save([dataX_pca,dataY],data_dir)
    perm = np.random.permutation(dataX_pca.shape[0])
    dataX_pca = dataX_pca[perm]
    dataY = dataY[perm]
    return dataX_pca[:dataSize,], dataY[:dataSize]
    
def filt_mnist_data(select_num):
    max_dataSize = 70000
    select_num = np.sort(select_num)
    remapped_num = np.arange(select_num.shape[0])
    dataX,dataY = generate_mnist_data(max_dataSize)
    tmp = np.full(max_dataSize, False, dtype=bool)
    for i in range(select_num.shape[0]):
        tmp = tmp | (dataY==select_num[i])
        dataY[dataY==select_num[i]] = remapped_num[i]
    dataX,dataY = dataX[tmp,:],dataY[tmp]
    return dataX,dataY
    
def generate_circle_data(dataSize):
    dataX, dataY = datasets.make_circles(n_samples=dataSize, shuffle=True, noise=0.2, random_state=None, factor=0.8)
    dataY = dataY
    return dataX, dataY
    
    
def generate_data(dataSize):
    return generate_mnist_data(dataSize)