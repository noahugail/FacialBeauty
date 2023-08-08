import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from keras_vggface.utils import preprocess_input

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
        
        p = np.random.permutation(len(batch_x))
        return batch_x[p], batch_y[p]
    
class Dataset():
    def __init__(
            self, 
            images,
            train_files,
            test_files,
            ratings,  
            input_shape, 
            batch_size,
            val_files=0
        ):

        self.mean = 0
        self.images = images
        self.ratings = pd.read_csv(ratings)
        self.input_shape = input_shape

        self.train_files = train_files
        self.test_files = test_files
        self.val_files = val_files

        self.batch_size = batch_size
        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(5/360),
            tf.keras.layers.RandomFlip("vertical")
        ])
    
    def shuffle(self, test_size=0.2, val_size=0.1):
        X_train, self.test.x, \
        y_train, self.test.y = train_test_split(
            np.concatenate((
                self.train.x[:len(self.train_files)],
                self.val.x,
                self.test.x
            )),
            np.concatenate((
                self.train.y[:len(self.train_files)],
                self.val.y,
                self.test.y
            )),
            shuffle=True,
            test_size=test_size
        )

        if not isinstance(self.val_files, int):
            X_train, self.val.x, \
            y_train, self.val.y = train_test_split(
                X_train,
                y_train,
                test_size=val_size
            )

        self.train.x, self.train.y = self.augment(X_train, y_train)

    def augment(self, X_train, y_train):
        if self.copies:
            X_train_copy = X_train.copy()
            y_train_copy = y_train.copy()
            for _ in range(self.copies):
                X_train = np.concatenate((X_train, self.augmentation(X_train_copy)))
                y_train = np.concatenate((y_train, y_train_copy))

        return X_train, y_train
    
    def preprocess(self, X, model):
        X = X[..., ::-1] #Convert from RGB to BGR

        #Zero center each channel w.r.t training dataset
        if isinstance(self.mean, int):
            self.mean = np.array([
                np.mean(X[..., 0]),
                np.mean(X[..., 1]),
                np.mean(X[..., 2])],
                np.float32
            )
        X -= self.mean
        
        if model: X = model.predict(X)
        return X

    def generate(self, model=None, augment=0):
        self.copies = augment

        X_train, y_train = self.load(self.train_files)
        X_test, y_test = self.load(self.test_files)

        X_train = self.preprocess(X_train, model)
        X_test = self.preprocess(X_test, model)

        X_train, y_train = self.augment(X_train, y_train)

        print(X_train.shape, X_test.shape)
        print(y_train.shape, y_test.shape)

        if not isinstance(self.val_files, int):
            X_val, y_val = self.load(self.val_files)
            X_val = self.preprocess(X_val, model)
            print(X_val.shape, y_val.shape)
            self.val = DataGenerator(X_val, y_val, self.batch_size)

        self.train = DataGenerator(X_train, y_train, self.batch_size)
        self.test = DataGenerator(X_test, y_test, self.batch_size)

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
    
def SCUTFBP5500(
        images="mediapipe/",
        directory="C:/Users/ugail/Downloads/SCUT-FBP5500_v2.1/SCUT-FBP5500_v2/",
        ratings="./SCUTFBP5500_Distribution.csv",
        input_shape=(224,224,3), 
        batch_size=32
    ):

        train_filenames = path_to_filenames(
            directory+"train_test_files/split_of_60%training and 40%testing/train.txt"
        )
        test_filenames = path_to_filenames(
            directory+"train_test_files/split_of_60%training and 40%testing/test.txt"
        )

        return Dataset(
            directory+images,
            train_filenames,
            test_filenames,
            ratings,
            input_shape,
            batch_size
        ), 5

def MEBeauty(
        images="cropped_images/images_crop_align_mtcnn3/",
        directory="C:/Users/ugail/Downloads/MEBeauty-database-main/MEBeauty-database-main/",
        ratings="./MEBeauty_Distribution.csv",
        input_shape=(224,224,3), 
        batch_size=32
    ):

        #filenames = pd.read_csv(ratings)["filename"].to_numpy()
        #train_files, test_files = split(filenames, 0.8)
        #train_files, val_files = split(train_files, 0.9)
        #print(len(train_files), len(test_files), len(val_files))

        train_files = path_to_filenames("./train_2023.txt")
        test_files = path_to_filenames("./test_2023.txt")
        val_files = path_to_filenames("./val_2023.txt")

        return Dataset(
            directory+images,
            train_files,
            test_files,
            ratings,
            input_shape,
            batch_size,
            val_files=val_files
        ), 10

def split(arr, percentage):
    idx = int(len(arr)*percentage)
    return arr[:idx], arr[idx:]
