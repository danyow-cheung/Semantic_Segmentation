'''we use the n-dim (for example 4-dim) softmax layer at the output layer to perform pixel-wise classification'''
from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Softmax, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import numpy as np

def build_fcn(input_shape,backone,n_classess = 4):
    '''Helper function to build an FCN model
    Arguments:
        backone(Model): A backone network such as ResNetv2 or v1
        n_classes(int): Number of object classes including background'''
    inputs = Input(shape = input_shape)
    features = backone(inputs)

    main_features = features[0]
    features = features[1:]
    out_features = [main_features]
    feature_size = 8 
    size = 2 

    # other half of the feature pyramid
    # including upsampling to restore the 
    # feature maps to the dimensions
    # equal to 1/4 the image size 

    for feature in features:
        postfix = 'fcn_'  + str(feature_size)
        feature = conv_layer(feature,filters = 256,use_maxpool=False,postfix=postfix)

        postfix = postfix+'_up2d'
        feature = UpSampling2D(size=size,interpolation='bilinear',name=postfix)(feature)

        size = size * 2 
        feature_size = feature_size * 2 
        out_features.append(feature)
    
    # concatenate all upsampled features
    x = Concatenate()(out_features)
    # perform 2 additional feature extraction and upsampling 
    x = tconv_layer(x,256,postfix='up_x2')
    x = tconv_layer(x,256,postfix="up_x4")
    # generate the pixel-wise classifier
    x = Conv2DTranspose(filters=n_classes,
                           kernel_size=1,
                           strides=1,
                           padding='same',
                           kernel_initializer='he_normal',
                           name="pre_activation")(x)