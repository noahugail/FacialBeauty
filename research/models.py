import tensorflow as tf
from tensorflow.keras import layers
from keras_vggface.vggface import VGGFace
from keras_vggface.models import resnet_identity_block, resnet_conv_block
from classification_models.tfkeras import Classifiers

"""
ResNet functions
"""
def Stage1_4(resnet):
    return tf.keras.models.Model(resnet.input, resnet.layers[-34].output)

def Stage5Unit1(resnet):
    return tf.keras.models.Model(resnet.layers[-33].input, resnet.layers[-22].output)

def Stage5(resnet):
    return tf.keras.models.Model(resnet.layers[-33].input, resnet.layers[-1].output)

def AvgPool7x7(input, output):
    return tf.keras.models.Model(input, layers.AveragePooling2D((7, 7))(output))

"""
Inception models
"""
def InceptionResNetV2(input_shape=(299, 299, 3)):
    base = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        input_shape=input_shape,
        pooling="None",
        weights="imagenet"
    )

    return base, None, tf.keras.applications.inception_resnet_v2.preprocess_input


"""
ResNeXt models
"""
def ResNeXtRB3(input_shape=(224, 224, 3)):
    ResNeXt50, preprocess_input = Classifiers.get("resnext50")
    resnext = ResNeXt50(include_top=False, input_shape=input_shape, weights="imagenet")

    layer1 = resnext.get_layer("stage3_unit6_relu")
    layer2 = resnext.get_layer("stage4_unit1_conv1")

    base = tf.keras.models.Model(resnext.input, layer1.output)
    trainable = AvgPool7x7(layer2.input, resnext.output)

    return base, trainable, preprocess_input

def ResNet41NeXt(input_shape=(224, 224, 3)):
    resnet = VGGFace(
        model="resnet50",
        include_top=False,
        input_shape=input_shape,
        weights="vggface"
    )
    base = Stage1_4(resnet)

    ResNeXt50, preprocess_input = Classifiers.get("resnext50")
    resnext = ResNeXt50(include_top=False, 
                        input_shape=input_shape, 
                        weights="imagenet")

    input = layers.Input(base.output_shape[1:])
    x = tf.keras.models.Model(resnext.get_layer("stage4_unit1_conv1").input,
                    resnext.output)(input)
    trainable = AvgPool7x7(input, x)

    return base, trainable, preprocess_input

def ResNet44NeXt(input_shape=(224, 224, 3)):
    resnet = VGGFace(
        model="resnet50",
        include_top=False,
        input_shape=input_shape,
        weights="vggface"
    )
    base = Stage1_4(resnet)

    ResNeXt50, preprocess_input = Classifiers.get("resnext50")
    resnext = ResNeXt50(include_top=False, 
                        input_shape=input_shape, 
                        weights="imagenet")

    input = layers.Input(base.output_shape[1:])
    x = Stage5Unit1(resnet)(input)
    x = tf.keras.models.Model(resnext.get_layer("stage4_unit2_conv1").input,
                    resnext.output)(x)
    trainable = AvgPool7x7(input, x)

    return base, trainable, preprocess_input


"""
ResNet models
"""
#ResNet50 minus last 6 convolutional layers with trainable Stage 5 Unit 1
def ResNet44(input_shape=(224, 224, 3)):
    resnet = VGGFace(
        model="resnet50",
        include_top=False,
        input_shape=input_shape,
        weights="vggface"
    )
    base = Stage1_4(resnet)

    input = layers.Input(base.output_shape[1:])
    x = Stage5Unit1(resnet)(input)
    trainable = AvgPool7x7(input, x)

    return base, trainable, tf.keras.applications.resnet50.preprocess_input

def ResNet44_IN(input_shape=(224, 224, 3)):
    resnet = tf.keras.applications.ResNet50(
        include_top=False, 
        input_shape=input_shape,
        weights="imagenet",
        pooling="None"
    )
    base = Stage1_4(resnet)

    input = layers.Input(base.output_shape[1:])
    x = Stage5Unit1(resnet)(input)
    trainable = AvgPool7x7(input, x)

    return base, trainable, tf.keras.applications.resnet50.preprocess_input

#ResNet50 minus last 9 convolutional layers
def ResNet41(input_shape=(224, 224, 3)):
    resnet = VGGFace(
        model="resnet50",
        include_top=False,
        input_shape=input_shape,
        weights="vggface"
    )
    base = Stage1_4(resnet)

    #input = layers.Input(base.output_shape[1:])
    #trainable = AvgPool7x7(input, input)
    
    return base, None, tf.keras.applications.resnet.preprocess_input

#ResNet50 with trainable Stage 5 with imagenet weights
def ResNetRB3_IN(input_shape=(224, 224, 3)):
    resnet = tf.keras.applications.ResNet50(
        include_top=False, 
        input_shape=input_shape,
        weights="imagenet",
        pooling="None"
    )
    base = Stage1_4(resnet)

    input = layers.Input(base.output_shape[1:])
    x = Stage5(resnet)(input)
    trainable = AvgPool7x7(input, x)

    return base, trainable, tf.keras.applications.resnet50.preprocess_input

#ResNet50 Stage 5 unfrozen
def ResNetRB3(input_shape=(224, 224, 3)):
    resnet = VGGFace(
        model="resnet50",
        include_top=False,
        input_shape=input_shape,
        weights="vggface"
    )
    base = Stage1_4(resnet)

    trainable = Stage5(resnet)

    return base, trainable, tf.keras.applications.resnet.preprocess_input

#ResNet50 Stage 5 unfrozen + Stage 4 Block 6 
def ResNetRB4(input_shape=(224, 224, 3)):
    resnet = VGGFace(
        model="resnet50",
        include_top=False,
        input_shape=input_shape,
        weights="vggface"
    )
    base = tf.keras.models.Model(resnet.input, outputs=resnet.layers[-44].output)

    trainable = tf.keras.models.Model(inputs=resnet.layers[-43].input, outputs=resnet.output)

    return base, trainable, tf.keras.applications.resnet.preprocess_input


"""
VGG16 models
"""
#ResNet Stage 1-4 + VGG trainable layers
def ResNet44VGG(input_shape=(224, 224, 3)):
    resnet = VGGFace(
        model="resnet50",
        include_top=False,
        input_shape=input_shape,
        weights="vggface"
    )
    base = Stage1_4(resnet)

    input = layers.Input(base.output_shape[1:])
    x = Stage5Unit1(resnet)(input)
    #x = tf.keras.layers.Conv2D(512, 3, strides=1, padding="same")(x)
    x = tf.keras.layers.Conv2D(512, 3, strides=1, padding="same")(x)
    x = tf.keras.layers.Conv2D(512, 3, strides=1, padding="same")(x)
    #x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)
    trainable = AvgPool7x7(input, x)

    return base, trainable, tf.keras.applications.resnet.preprocess_input


def VGG16(input_shape=(224, 224, 3)):
    vgg = VGGFace(
        model="vgg16",
        include_top=False,
        input_shape=input_shape,
        weights="vggface"
    )
    base = tf.keras.models.Model(vgg.input, vgg.get_layer("pool4").output)
   
    input = layers.Input(base.output_shape[1:])
    x = tf.keras.models.Model(vgg.get_layer("conv5_1").input, vgg.output)(input)
    trainable = AvgPool7x7(input, x)

    return base, trainable, tf.keras.applications.vgg16.preprocess_input


"""
SqueezeNet models
"""
#Senet + ResNet Stage 5
def SEResNet(input_shape=(224, 224, 3)):
    resnet = VGGFace(
        model="resnet50",
        include_top=False,
        input_shape=input_shape,
        pooling="None",
        weights="vggface"
    )

    senet = VGGFace(
        model="senet50",
        include_top=False,
        input_shape=input_shape,
        pooling="None",
        weights="vggface"
    )

    base = tf.keras.models.Model(inputs=senet.input, outputs=senet.layers[-55].output)
    trainable = Stage5(resnet)

    return base, trainable, tf.keras.applications.resnet.preprocess_input


"""
DenseNet models
"""
#ResNet Stage 1-4 + densenet Conv layers
def ResNet41Dense(input_shape=(224, 224, 3)):
    resnet = VGGFace(
        model="resnet50",
        include_top=False,
        input_shape=input_shape,
        pooling="None",
        weights="vggface"
    )
    base = Stage1_4(resnet)

    densenet = tf.keras.applications.DenseNet201(include_top=False,
            input_shape=input_shape,
            weights="imagenet"
    )
    densenet = tf.keras.models.Model(densenet.get_layer("pool4_pool").input,
                densenet.output)

    inputs = layers.Input(base.output_shape[1:])
    x = tf.keras.layers.Conv2D(896, 1, strides=1)(inputs)
    x = densenet(x)
    trainable = tf.keras.models.Model(inputs, x)

    return base, trainable

def DenseNet201(input_shape=(224, 224, 3)):
    base = tf.keras.applications.DenseNet201(include_top=False,
            input_shape=input_shape,
            weights="imagenet",
            pooling="None"
    )
    return base, None, tf.keras.applications.densenet.preprocess_input

def DenseNet169(input_shape=(224, 224, 3)):
    base = tf.keras.applications.DenseNet169(include_top=False,
            input_shape=input_shape,
            weights="imagenet",
            pooling=None
    )

    """
    base = tf.keras.models.Model(
        densenet.input,
        densenet.get_layer("conv5_block1_0_relu").output
    )
   
    input = layers.Input(base.output_shape[1:])
    print(input)
    x = tf.keras.models.Model(
        densenet.get_layer("conv5_block1_1_conv").input,
        densenet.get_layer("relu").output
    )(input)
    trainable = AvgPool7x7(input, x)
    """

    return base, None, tf.keras.applications.densenet.preprocess_input