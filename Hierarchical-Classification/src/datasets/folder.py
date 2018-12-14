import os
import sys
import torch
from torch.utils.data import Dataset
from utils import list_dir
from .utils import default_loader, make_dataset
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

        
class DatasetFolder(Dataset):
    def __init__(self, root, loader, extensions, transform=None):
        dirs = list_dir(root)
        if(dirs != []):
            self.if_classify = True
            self.classes, self.classes_to_labels = self._find_classes(root)
            self.classes_size = len(self.classes_to_labels.keys())
        else:
            self.if_classify = False
            self.classes_to_labels = None
        samples = make_dataset(root, extensions, self.classes_to_labels)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        self.root = root
        self.loader = loader
        self.extensions = extensions       
        self.samples = samples
        self.transform = transform

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        classes_to_labels = {classes[i]: i for i in range(len(classes))}
        return classes, classes_to_labels

    def __getitem__(self, index):
        if(self.if_classify):
            path, label = self.samples[index]
            label = torch.tensor(label)
            img = self.loader(path)
            input = {'img': img, 'label': label}
        else:
            path = self.samples[index]
            img = self.loader(path)
            input = {'img': img}            
        if self.transform is not None:
            input = self.transform(input)            
        return input

    def __len__(self):
        return len(self.samples)
        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform)