import tensorflow as tf
import numpy as np
import pickle
import os
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import backend as K

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(1234)

import datasets
import losses
import models
from build import build, train

base, trainable, preprocess_input = models.ResNet41()
path = "C:/Users/ugail/Documents/research/"
n = 5

if not os.path.exists(path+"dataset.pkl"):
    dataset, _ = datasets.SCUTFBP5500(input_shape=base.input_shape)
    dataset.generate(model=base, preprocess_input=preprocess_input, augment=0)

    with open(path+"dataset.pkl", 'wb') as p:
        pickle.dump(dataset, p, pickle.HIGHEST_PROTOCOL)

else:
    with open(path+"dataset.pkl", 'rb') as inp:
        dataset = pickle.load(inp)

n1 = 0
if not os.path.exists(path+"result.csv"):
    df = pd.DataFrame(0,
        index=np.arange(3),
        columns=["loss","mae","rmse","pc",
                "mae2","rmse2","pc2",
                "time,","itr"]
    )

else:
    df = pd.read_csv(path+"result.csv")
    n1 = int(np.round(df.iloc[0,-1]))

#n2 = np.load(path+"n2.npy")[0]
n2 = 50

for i in range(n1,n2):
    tf.keras.backend.clear_session()
    model = build(
        base.output_shape,
        trainable=trainable,
        n=n,
        augment=False,
    )

    loss = losses.SquaredEarthMoversDistance()
    metrics = [losses.MeanAbsoluteError(n=n),
            losses.RootMeanSquaredError(n=n),
            losses.PearsonCorrelation(n=n),
            losses.MeanAbsoluteErrorSD(n=n),
            losses.RootMeanSquaredErrorSD(n=n),
            losses.PearsonCorrelationSD(n=n),]

    time = train(model, loss, dataset.train, dataset.test, metrics, verbose=0)
    result = model.evaluate(dataset.test, verbose=0)
    print()
    print(f"{i+1}/{n2}")
    print(result)
    print()

    itr = df.iloc[0,-1]+1
    df.iloc[0,:-2] += np.asarray(result)
    df.iloc[0,-2] += time
    df.iloc[1,:-1] = df.iloc[0,:-1]/itr

    df.iloc[0,-1] = itr
    df.iloc[1,-1] = itr
    if result[3] > df.iloc[2,3]:
        df.iloc[2,:-2] = result
        df.iloc[2,-2] = time
        df.iloc[2,-1] = itr

    df.to_csv(path+"result.csv", sep=",", index=False)

