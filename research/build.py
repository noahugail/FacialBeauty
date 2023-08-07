import tensorflow as tf
from tensorflow.keras import layers
from livelossplot import PlotLossesKerasTF
import timeit
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.keras.activations.softmax

taylor = lambda x : 1+x+(0.5)*(x**2)#+(1/6)*(x**3)+(1/24)*(x**3)
def softmax_taylor(x):
    e = taylor(x - tf.reduce_max(x, keepdims=True))
    s = tf.reduce_sum(e, keepdims=True)
    return e / s

def build(
        input_shape,
        train=0,
        n=5,
        base=None, 
        trainable=None, 
        mlp=None, 
        augment=False, 
    ):

    if base:
        for l in base.layers[:int(len(base.layers)*(1-train))]:
            l.trainable = False
    
    if not mlp:
        mlp = [
            layers.Flatten(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
        ]

    head = layers.Dense(n, activation="softmax")

    seq = [
        *mlp,
        head
    ]

    #if trainable: seq.insert(0, tf.keras.models.clone_model(trainable))
    if trainable: seq.insert(0, trainable)
    if base: seq.insert(0, base)
    if augment: seq.insert(0, layers.RandomRotation(5/360))

    model = tf.keras.Sequential(seq)
    if base:
        model.build(base.input_shape)
    else:
        model.build(input_shape)

    model.summary()

    return model

def compile(model, loss, metrics=None, learning_rate=0.0006):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )

def train(
        model, train, test, 
        callbacks=None, epochs=1000, patience=50, verbose=1,
        monitor="val_correlation", folder="./test/",
    ):

    mode = "max" if monitor == "val_correlation" else "min"

    print(folder)
    stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        mode=mode
    )

    if callbacks:
        callbacks += [stopping]
    else:
        callbacks = [stopping]

    start = timeit.default_timer()
    model.fit(
        train,
        epochs=epochs,
        validation_data=test,
        callbacks=callbacks,
        verbose=verbose
    )

    return (timeit.default_timer() - start)/60