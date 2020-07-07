import os
import numpy as np
import pandas as pd
import argparse
import gzip, pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import sklearn
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, AlphaDropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import to_categorical
from keras.regularizers import l2



# Creating an instance of the ArgumentParser object
ap = argparse.ArgumentParser()

# Adding arguments that one would want the user to enter while running the program from shell or terminal
ap.add_argument("-l", "--lr", required = True, help = "Learning rate")
ap.add_argument("-b", "--batch_size", required = True, help = "Batch size - valid values are 1 or multiples of 5")
ap.add_argument("-i", "--init", required = True, help = "Initialization method - Enter 1 for Xavier and 2 for He")
ap.add_argument("-sd", "--save_dir", required = True, help = "path to the directory where the model should get saved (i.e) all weights and biases")
ap.add_argument("-te", "--train_or_eval", required = True, help = "Enter 1 if you want to train the model from scratch or enter 2 to load the best model")
ap.add_argument("-lm", "--load_model_dir", help = "enter the path of the model that you want to load")

# Dictionary containing all the arguments
args = vars(ap.parse_args())

# Extracting the argument values
lr = float(args["lr"])
init = int(args["init"])
batch_size = int(args["batch_size"])
save_dir = str(args["save_dir"])
t_eval = int(args["train_or_eval"])
lm = str(args["load_model_dir"])

if init == 1:
    init = "he_normal"
elif init == 2:
    init = "GlororNormal"


if t_eval == 1:

    # Function to load the pickled format of the file and then concatenating all the 5 batches into a single batch 
    def load_data():
        # Setting seed to a some value so as to have the same random distribution while fitting
        tf.random.set_seed(123)
        np.random.seed(123)

        os.chdir(os.path.join(os.curdir, "cifar-10-batches-py/"))

        # Function to unpickle the pickled file
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        # Unpickling every batch which are all dictionaries
        data_batch_1 = unpickle('data_batch_1')
        data_batch_2 = unpickle('data_batch_2')
        data_batch_3 = unpickle('data_batch_3')
        data_batch_4 = unpickle('data_batch_4')
        data_batch_5 = unpickle('data_batch_5')
        test_batch = unpickle('test_batch')
        class_names = unpickle('batches.meta')

        # Extracting the train, test batches and class_names from the dictionaries
        train_data_1 = data_batch_1[b'data']
        train_data_2 = data_batch_2[b'data']
        train_data_3 = data_batch_3[b'data']
        train_data_4 = data_batch_4[b'data']
        train_data_5 = data_batch_5[b'data']
        test_data = test_batch[b'data']

        train_lab_1 = data_batch_1[b'labels']
        train_lab_2 = data_batch_2[b'labels']
        train_lab_3 = data_batch_3[b'labels']
        train_lab_4 = data_batch_4[b'labels']
        train_lab_5 = data_batch_5[b'labels']
        test_lab = test_batch[b'labels']

        class_names = class_names[b"label_names"]

        # Concatenating all the train and test batches into a single training and a single test image set respectively
        train_data = tf.concat([train_data_1,train_data_2,train_data_3,train_data_4,train_data_5], 0) 
        train_labels = tf.concat([train_lab_1,train_lab_2,train_lab_3,train_lab_4,train_lab_5], 0)

        return train_data, test_data, train_labels, test_lab, class_names


    # Function to preprocess the data loaded by load_data()
    def preprocess_data():
        
        # Loads data from load_data() function
        train_data, test_data, train_labels, test_lab, class_names = load_data()

        # Function to resize the image data from m*3072 to m*32*32*3 (where m is 50000 for train set and 10000 for test set)
        # We need some workaround here, because the 3072 elements contain 1024 elements each for red, green, blue channels in that order.
        def resize(data):
            data1 = tf.reshape(data, [-1,3,32,32]).numpy()
            data2 = tf.transpose(data1, perm = [0,2,3,1]).numpy()
            return data2

        # Reshaping train and test images to appropriate shape
        train_images = resize(train_data)
        test_data = resize(test_data)

        # Setting seed to a some value so as to have the same random distribution while fitting
        tf.random.set_seed(123)
        np.random.seed(123)

        X = np.array(train_images)
        y = np.array(train_labels)

        # Using StratifiedShuffleSplit from scikit-learn by first creating its instance and then splitting the train set further into train and validation set
        # 45000 images in train set and 5000 images in validation set
        # Stratified sampling is better than random sampling because this takes care of skewed classes
        split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.1, random_state = 123)
        for train_index, val_index in split.split(X,y):
            X_train = X[train_index]
            X_valid = X[val_index]
            y_train = y[train_index]
            y_valid = y[val_index]

        X_test, y_test = np.array(test_data), np.array(test_lab)
        X_train, X_valid, X_test = X_train/255.0, X_valid/255.0, X_test/255.0

        y_train = to_categorical(y_train, num_classes = 10)
        y_valid = to_categorical(y_valid, num_classes = 10)
        y_test = to_categorical(y_test, num_classes = 10)

        return X_train, y_train, X_valid, y_valid, X_test, y_test, class_names    

    #Calling the preprocess_data() function to preprocess the data
    X_train, y_train, X_valid, y_valid, X_test, y_test, class_names = preprocess_data()

    #Visualizing 25 random images from the train set and its corresponding label
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i])
        plt.xlabel(class_names[np.argmax(y_train[i])].decode("utf-8"))
    plt.show()


    learning_rate = lr
    init = init
    reg_rate = 1e-3  # This regularization rate seems to give good result, you could also use 5e-4

    # Setting seed to a some value so as to have the same random distribution
    tf.random.set_seed(123)
    np.random.seed(123)

    model = Sequential([

                Conv2D(64, 3, strides = 1, padding = "same", activation = "relu", kernel_initializer = init, 
                       bias_initializer='zeros', kernel_regularizer = l2(reg_rate), input_shape = [32,32,3]),

                BatchNormalization(),

                MaxPool2D(pool_size=2, strides = 2),

                Conv2D(128, 3, strides = 1, padding = "same", activation = "relu", kernel_initializer = init, 
                       bias_initializer='zeros', kernel_regularizer = l2(reg_rate)),

                BatchNormalization(),          

                MaxPool2D(pool_size=2, strides = 2),

                Conv2D(256, 3, strides = 1, padding = "same", activation = "relu", kernel_initializer = init, 
                       bias_initializer='zeros', kernel_regularizer = l2(reg_rate)),

                Conv2D(256, 3, strides = 1, padding = "same", activation = "relu", kernel_initializer = init, 
                       bias_initializer='zeros', kernel_regularizer = l2(reg_rate)),

                BatchNormalization(),

                MaxPool2D(pool_size=2, strides = 2),

                    Flatten(),
            
                Dense(1024, activation = "relu", kernel_initializer = init, bias_initializer='zeros', 
                      kernel_regularizer = l2(reg_rate)),

                BatchNormalization(),

                AlphaDropout(0.5),

                Dense(1024, activation = "relu", kernel_initializer = init, bias_initializer='zeros', 
                      kernel_regularizer = l2(reg_rate)),

                BatchNormalization(),

                AlphaDropout(0.5),          

                Dense(10, activation="softmax"), 
    ])
    opt = keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

    # Summary of the compiled Model
    model.summary()

    # Defining a root directory for saving tensorboard files 
    root_logdir = os.path.join(os.curdir, "my_logs")

    # Function for Creating a directory to contain the TensorBoard files based on the run time of the model
    def get_run_logdir(log_dir):
        import time
        run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
        return os.path.join(log_dir, run_id)

    # Function for Creating a directory to contain the saved model based on the run time of the model
    def get_model_logdir(log_dir):
        import time
        model_id = time.strftime("model_%Y_%m_%d_%H_%M_%S")
        return os.path.join(log_dir, model_id)
    
    # Creating a directory 
    run_logdir = get_run_logdir(root_logdir)

    # Defining Callbacks that need to be used to monitor the val_accuracy and TensorBoard to visualize the graphs
    early_stopping_cb = EarlyStopping(monitor="val_accuracy", min_delta=0, patience=5, verbose=1, mode='auto', restore_best_weights = True)
    lr_scheduler = ReduceLROnPlateau(factor = 0.5, patience = 4)
    tensorboard_cb = TensorBoard(run_logdir)

    # Setting seed to a some value so as to have the same random distribution while fitting
    tf.random.set_seed(123)
    np.random.seed(123)
    history = model.fit(X_train, y_train, epochs = 100, batch_size = batch_size, validation_data = (X_valid, y_valid), 
                               callbacks = [early_stopping_cb, lr_scheduler, tensorboard_cb])

    # To visualize the learning curve 
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,5)
    plt.xlabel("Epochs")
    plt.show()

    # Evaluation of the model trained from scratch, on the test set
    model.evaluate(X_test, y_test)
    
    # Creating a directory to save the trained models with it's time of training
    path = os.path.join(save_dir, "saved_models")
    model_logdir = get_model_logdir(path)
    save_model(model, model_logdir, save_format = 'h5')

elif t_eval == 2:
    # Loading the best model which was saved from training previously using the argument lm
    model1 =  load_model(lm)
    # Evaluation of the best model on the test set
    model1.evaluate(X_test, y_test)

    # model.save("prog_assign_2(82.63).h5")


