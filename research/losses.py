import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import numpy as np

"""
Softmax losses
"""
def SquaredEarthMoversDistance():
    def squared_earth_movers_distance(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        cdf_true = K.cumsum(y_true, axis=-1)
        cdf_pred = K.cumsum(y_pred, axis=-1)
        return K.mean(K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1)))
    
    return squared_earth_movers_distance
    

"""
Mean losses
"""
def MeanAbsoluteError(n=5):
    def multiply(y):
        return y*(n*tf.constant(np.arange(1,n+1), dtype=tf.float32))
     
    def mean(y):
        return tf.math.reduce_mean(multiply(y), axis=1)

    def mean_absolute_error(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_mean = mean(y_true)
        y_pred_mean = mean(y_pred)
        return tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))

    return mean_absolute_error

def RootMeanSquaredError(n=5):
    def multiply(y):
        return y*(n*tf.constant(np.arange(1,n+1), dtype=tf.float32))
     
    def mean(y):
        return tf.math.reduce_mean(multiply(y), axis=1)
    
    def root_mean_squared_error(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_mean = mean(y_true)
        y_pred_mean = mean(y_pred)
        return tf.sqrt(tf.reduce_mean(tf.square(y_true_mean - y_pred_mean)))
    
    return root_mean_squared_error

def PearsonCorrelation(n=5):
    def multiply(y):
        return y*(n*tf.constant(np.arange(1,n+1), dtype=tf.float32))
     
    def mean(y):
        return tf.reshape(tf.math.reduce_mean(multiply(y), axis=1), [-1, 1])
    
    def correlation(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_mean = mean(y_true)
        y_pred_mean = mean(y_pred)
        return tfp.stats.correlation(y_true_mean, y_pred_mean)[0][0]
    
    return correlation


"""
Standard deviation losses
"""
def MeanAbsoluteErrorSD(n=5):
    def multiply(y):
        return y*(n*tf.constant(np.arange(1,n+1), dtype=tf.float32))
    
    def std(y):
        return tf.math.reduce_std(multiply(y), axis=1)
    
    def mean_absolute_error(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_std = std(y_true)
        y_pred_std = std(y_pred)
        return tf.reduce_mean(tf.abs(y_true_std - y_pred_std))

    return mean_absolute_error

def RootMeanSquaredErrorSD(n=5):
    def multiply(y):
        return y*(n*tf.constant(np.arange(1,n+1), dtype=tf.float32))
    
    def std(y):
        return tf.math.reduce_std(multiply(y), axis=1)
    
    def root_mean_squared_error(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_std = std(y_true)
        y_pred_std = std(y_pred)
        return tf.sqrt(tf.reduce_mean(tf.square(y_true_std - y_pred_std)))
    
    return root_mean_squared_error

def PearsonCorrelationSD(n=5):
    def multiply(y):
        return y*(n*tf.constant(np.arange(1,n+1), dtype=tf.float32))
     
    def std(y):
        return tf.reshape(tf.math.reduce_std(multiply(y), axis=1), [-1, 1])
    
    def correlation(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_std = std(y_true)
        y_pred_std = std(y_pred)
        return tfp.stats.correlation(y_true_std, y_pred_std)[0][0]
    
    return correlation