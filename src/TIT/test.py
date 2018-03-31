import numpy as np
import theano
import theano.tensor as T
import gtic

def compile():
    x = T.tensor3()
    return theano.function([x], x.flatten(2))


def main():
    a = [np.array([[1,2],[2,3],[3,4]]),np.array([[1],[2],[3]])]
    print(a[0].shape)
    gtic.TIC.TIC()
    
main()