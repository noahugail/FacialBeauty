import tensorflow as tf
#import tensorflow_addons as tfa
from livelossplot import PlotLossesKerasTF
import timeit
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def build(
        input_shape,
        train=0,
        n=5,
        base=None, 
        trainable=None, 
        mlp_weights=None, 
        augment=False, 
    ):

    if base:
        for layer in base.layers[:int(len(base.layers)*(1-train))]:
            layer.trainable = False

    inputs = tf.keras.layers.Input(
        trainable.output_shape[1:] if trainable else input_shape[1:]
    )
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(n, activation="softmax")(x)
    
    mlp = tf.keras.Model(inputs, outputs)

    if not mlp_weights:
        return mlp
    
    mlp.load_weights(mlp_weights)
    #for layer in mlp.layers:
        #layer.trainable = False

    inputs = tf.keras.layers.Input(
        base.input_shape[1:] if base else input_shape[1:]
    )
    x = inputs
    if augment: x = tf.keras.layers.RandomRotation(0.1)(x) #5/360
    if base: x = base(x)#, training=False)
    if trainable: x = trainable(x)#, training=False)
    outputs = mlp(x)

    model = tf.keras.Model(inputs, outputs)
    #if trainable: seq.insert(0, tf.keras.models.clone_model(trainable))
    model.summary()

    return model

def compile(model, loss, metrics=None, learning_rate=0.0006):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                       #tfa.optimizers.AdamW(weight_decay=5e-5),
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