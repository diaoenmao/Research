import pickle

def save(input, dir):
    pickle.dump(input, open(dir, "wb" ))
    return

def load(dir):
    return pickle.load(open(dir, "rb" ))