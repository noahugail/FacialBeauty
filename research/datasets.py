import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
    
class Dataset():
    def __init__(self, 
                images,
                train_files,
                test_files,
                ratings,  
                input_shape, 
                batch_size):
        
        self.images = images
        self.ratings = pd.read_csv(ratings)
        self.input_shape = input_shape
        self.train_files = train_files
        self.test_files = test_files
        self.batch_size = batch_size

    def generate(self, model=None, preprocess_input=None, augment=0):
        X_train, y_train = self.load(self.train_files)
        X_test, y_test = self.load(self.test_files)

        self.X_train = X_train

        if preprocess_input:
            X_train = preprocess_input(X_train)
            X_test = preprocess_input(X_test)

        if model:
            X_train = model.predict(X_train)
            X_test = model.predict(X_test)

        augmentation = tf.keras.Sequential([
            layers.RandomRotation(5/360),
            layers.RandomFlip("vertical")
        ])

        X_train_copy = X_train.copy()
        y_train_copy = y_train.copy()
        for _ in range(augment):
            X_train = np.concatenate((X_train, augmentation(X_train_copy)))
            y_train = np.concatenate((y_train, y_train_copy))

        del X_train_copy, y_train_copy

        print(X_train.shape, X_test.shape)
        print(y_train.shape, y_test.shape)

        self.train = DataGenerator(X_train, y_train, self.batch_size)
        self.test = DataGenerator(X_test, y_test, self.batch_size)

        del X_train, X_test, y_train, y_test

    def load(self, files):
        X = []
        y = []
        for file in files:
            if os.path.exists(self.images+file) and file in self.ratings["filename"].to_numpy():
                label = np.asarray(
                            self.ratings.loc[self.ratings["filename"]==file]
                            .to_numpy()[0][2:-1], 
                            dtype=np.float32
                    )
                label /= np.sum(label)
                y.append(label)

                X.append(img_to_array(load_img(
                        self.images+file, target_size=(self.input_shape[1], 
                                            self.input_shape[2])
                )))

        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32)

        return X, y
    
def path_to_filenames(path):
        with open(path, "r") as f:
            return [l.split(".jpg")[0]+".jpg" for l in f.readlines()]
    
def SCUTFBP5500(images="mediapipe/",
                directory="C:/Users/ugail/Downloads/SCUT-FBP5500_v2.1/SCUT-FBP5500_v2/",
                ratings="./SCUTFBP5500_Distribution.csv",
                input_shape=(224,224,3), 
                batch_size=32):

        train_filenames = path_to_filenames(
            directory+"train_test_files/split_of_60%training and 40%testing/train.txt"
        )
        test_filenames = path_to_filenames(
            directory+"train_test_files/split_of_60%training and 40%testing/test.txt"
        )

        return Dataset(directory+images,
                       train_filenames,
                       test_filenames,
                       ratings,
                       input_shape,
                       batch_size), 5

def MEBeauty(images="cropped_images/images_crop_align_mtcnn2/",
            directory="C:/Users/ugail/Downloads/MEBeauty-database-main/MEBeauty-database-main/",
            ratings="./MEBeauty_Distribution.csv",
            input_shape=(224,224,3), 
            batch_size=32):

        
        filenames = pd.read_csv(ratings)["filename"].to_numpy()

        train_filenames, test_filenames = train_test_split(filenames,
                                                           shuffle=True,
                                                           test_size=0.2
                                        )
        
        train_filenames, _, = train_test_split(train_filenames, 
                                               test_size=0.1
                            )
        
        """
        train_filenames = path_to_filenames(
            directory+"scores/train_2023.txt"
        )
        test_filenames = path_to_filenames(
            directory+"scores/test_2023.txt"
        )
        """

        return Dataset(directory+images,
                       train_filenames,
                       test_filenames,
                       ratings,
                       input_shape,
                       batch_size), 10
