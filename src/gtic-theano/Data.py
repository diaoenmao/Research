from sklearn import datasets

def generate_circle_data(dataSize):
    dataX, dataY = datasets.make_circles(n_samples=dataSize, shuffle=True, noise=0.2, random_state=None, factor=0.6)
    dataY = dataY.reshape((-1,1))
    dim_out = 2
    return dataX, dataY, dim_out
    
    
def generate_data(dataSize):
    return generate_circle_data(dataSize)