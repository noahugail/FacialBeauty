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
ResNet models
"""
#ResNet50 VGGFace2 weights minus last 6 convolutional layers with trainable Stage 5 Unit 1
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

    return base, trainable

#ResNet50 ImageNet weights minus last 6 convolutional layers with trainable Stage 5 Unit 1
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

    return base, trainable

#ResNet50 with trainable Stage 5 with imagenet weights
def ResNet50_IN(input_shape=(224, 224, 3)):
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

    return base, trainable

#ResNet50 VGGFace2 weights trainable Stage 5
def ResNet50(input_shape=(224, 224, 3)):
    resnet = VGGFace(
        model="resnet50",
        include_top=False,
        input_shape=input_shape,
        weights="vggface"
    )
    base = Stage1_4(resnet)

    trainable = Stage5(resnet)

    return base, trainable
    

"""
VGG16 models
"""
#VGG16 with trainable Stage 5 with vggface weights
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

    return base, trainable

#VGG16 with trainable Stage 5 with imagenet weights
def VGG16_IN(input_shape=(224, 224, 3)):
    vgg = tf.keras.applications.VGG16(
        include_top=False,
        input_shape=input_shape,
        weights="imagenet"
    )
    base = tf.keras.models.Model(vgg.input, vgg.get_layer("block4_pool").output)
   
    input = layers.Input(base.output_shape[1:])
    x = tf.keras.models.Model(vgg.get_layer("block5_conv1").input, vgg.output)(input)
    trainable = AvgPool7x7(input, x)

    return base, trainable
