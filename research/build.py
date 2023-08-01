import tensorflow as tf
from tensorflow.keras import layers
from livelossplot import PlotLossesKerasTF
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
        for l in base.layers[:len(base.layers)*(1-train)]:
            l.trainable = False
    
    if not mlp:
        mlp = [
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
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

def train(
        model, loss, train, test, metrics, 
        callbacks=None, epochs=1000, patience=50, learning_rate=0.0006,
        monitor="val_correlation", folder="./test/",
    ):

    mode = "max" if monitor == "val_correlation" else "min"

    print(folder)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )

    stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        mode=mode
    )

    if callbacks:
        callbacks += [PlotLossesKerasTF(), stopping]
    else:
        callbacks = [PlotLossesKerasTF(), stopping]

    model.fit(
        train,
        epochs=epochs,
        validation_data=test,
        callbacks=callbacks
    )