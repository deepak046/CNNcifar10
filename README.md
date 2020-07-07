# CNNcifar10
## Classification using CNN on CIFAR-10 dataset

Download the CIFAR-10 dataset from http://www.cs.toronto.edu/~kriz/cifar.html

### Something about the dataset

The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch. Each of these files is a Python **"pickled"** object produced with **cPickle**. Here is a python3 routine which will open such a file and return a dictionary: 

```
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
```
The files should be unpickled inside the workspace
