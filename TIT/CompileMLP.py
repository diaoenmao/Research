from MLP import *
import numpy as np
import copy
import theano
from theano.misc.pkl_utils import dump, load, StripPickler
import pickle
DIM_IN = 784
DIM_OUT = 10

MIN_DIM_HIDDEN = 1
MAX_DIM_HIDDEN = 10
MIN_DEPTH_HIDDEN = 1
MAX_DEPTH_HIDDEN = 3

output_dir = 'TIC_MLP.pkl'

def CompileMLP(dim_in, min_dim_hidden, max_dim_hidden, min_depth_hidden, max_depth_hidden, dim_out, cur_depth_hidden, prev_dim_hidden):
    if(cur_depth_hidden == max_depth_hidden):
        print(("Current Layer: %d" % cur_depth_hidden))
        for i in range(min_dim_hidden,max_dim_hidden+1):
            print(("Current node: %d" % i))
            if(i==min_dim_hidden):
                prev_dim_hidden.append(i)  
            else:
                prev_dim_hidden[-1] = i
            cur_dim_models = BuildMLP(dim_in, prev_dim_hidden, dim_out)
            print(("Compiled for %s\n" % prev_dim_hidden))
            pickle.dump(cur_dim_models, open(output_dir, "ab" ))
        return
    else:
        print(("Current Layer: %d" % cur_depth_hidden))
        for i in range(min_dim_hidden,max_dim_hidden+1):
            print(("Current node: %d" % i))
            if(i==min_dim_hidden):
                prev_dim_hidden.append(i)  
            else:
                prev_dim_hidden[-1] = i                
            cur_dim_models = BuildMLP(dim_in, prev_dim_hidden, dim_out)
            print(("Compiled for %s\n" % prev_dim_hidden))
            CompileMLP(dim_in, min_dim_hidden, max_dim_hidden, min_depth_hidden, max_depth_hidden, dim_out, cur_depth_hidden+1, copy.deepcopy(prev_dim_hidden))
            pickle.dump(cur_dim_models, open(output_dir, "ab" ))
    return

def main():
    CompileMLP(DIM_IN, MIN_DIM_HIDDEN, MAX_DIM_HIDDEN, MIN_DEPTH_HIDDEN, MAX_DEPTH_HIDDEN, DIM_OUT, MIN_DEPTH_HIDDEN, [])

if __name__ == "__main__":
    main()     
