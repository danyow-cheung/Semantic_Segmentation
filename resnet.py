'''build the features' pyramid and the unsampling and prediction layers'''
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.layers import Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import numpy as np

from model import conv_layer
def features_pyramid(x,n_layers):
    '''Generate features pyramid from the output of the last layer of a backone network(eg ResNetv1 or v2)
    
    Arguments:
        x(tensor):          output feature maps of a backone network
        n_layers(int):      Number of additional pyramid layers
    
    Return:
        outputs(list):      Feature pyramid
    
    '''
    outputs = [x]
    conv = AveragePooling2D(pool_size=2,name='pool1')(x)
    outputs.append(conv)
    prev_conv = conv 
    n_filters = 512 
    # additional feature map layers
    for i in range(n_layers-1):
        postfix = '_layer'+str(i+2)
        conv = conv_layer(prev_conv,n_filters,kernel_size=3,strides=2,use_maxpool=False,postfix=postfix)
        outputs.append(conv)
        prev_conv = conv 
    return outputs


