'''

Given the segmentation network model, 
we use the Adam optimizer with a learning rate of 1e-3 
and a categorical cross-entropy loss function to train the network
'''
'''
The learning rate is halved every 20 epochs after 40 epochs. 
'''
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler

import os
import skimage
import numpy as np

from data_generator import DataGenerator
from model_utils import parser, lr_scheduler
os.sys.path.append("../lib")
from common_utils import print_log, AccuracyCallback
from model import build_fcn
from skimage.io import imread




class FCN:
    """Made of an fcn model and a dataset generator.
    Define functions to train and validate an FCN model.
    Arguments:
        args: User-defined configurations
    Attributes:
        fcn (model): FCN network model
        train_generator: Multi-threaded 
            data generator for training
    """
    def __init__(self, args):
        """Copy user-defined configs.
        Build backbone and fcn network models.
        """
        self.args = args
        self.fcn = None
        self.train_generator = DataGenerator(args)
        self.build_model()
        self.eval_init()

    def build_model(self):
        '''build a backone network and use it to create a semantic segmentation
        network based on FCN 
        '''
        # input shape is (480,640,3) by default 
        self.input_shape = (self.args.height,self.args.width,self.args.channels)

        # build the backone network (eg ResNet50)
        # the backone is used for 1sr set of features
        # of the features pyramid 
        self.backone = self.args.backone(self.input_shape,n_layers = self.args.layers)

        # using the backone .build fcnn network
        # output layer is a pixel-wise classifier
        self.n_classes = self.train_generator.n_classes
        self.fcn = build_fcn(self.input_shape,self.backone,self.n_classes)

    def train(self):
        '''Train an FCN'''
        optimizer = Adam(lr=1e-3)
        loss = 'categorical_crossentropy'
        self.fcn.compile(optimizer=optimizer, loss=loss)

        log = '# of classes %d '%self.n_classes

        print_log(log,self.args.verbose)
        log = "Batch size :%d"%self.args.batch_size 
        print_log(log,self.args.verbose)

        # prepare callbacks for saving model weights
        # and learning rate scheduler 
        # model weights are saved when test iou is highest
        # learning rate decreases by 50% every 20 epochs 
        # after 40th epoch 
        accuracy = AccuracyCallback(self)
        scheduler = LearningRateScheduler(lr_scheduler)

        callbacks = [accuracy,scheduler]
        # train the fcn network
        self.fcn.fit_generator(generator=self.train_generator,
                                use_multiprocessing=True,
                                callbacks=callbacks,
                                epochs=self.args.epochs,
                                workers =self.args.workers)
        

    def eval_init(self):
        '''Housekeeping for trained model evaluation'''
        # model weights are saved for future validation
        # prepare model ,model saving directory
        save_dir = os.path.join(os.getcwd(),self.args.save_dir)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        model_name = self.backone.name 
        model_name += '-' + str(self.args.layers)+'layer-'
        model_name += self.args.dataset
        model_name += '-best-iou.h5'
        log = "Weights filename :%s"%model_name
        print_log(log,self.args.verbose)
        self.weights_path = os.path.join(save_dir,model_name)
        self.preload_test()
        self.miou = 0 
        self.miou_history = []
        self.mpla_history= []
    
    def preload_test(self):
        '''pre-load test dataset to save time'''
        path = os.path.join(self.args.data_path,self.args.test_labels)

        # ground truth data is stored in an npy file 
        self.test_dictionary = np.load(path,allow_pickle=True).flat[0]

        self.test_keys = n 
        print_log("Loaded %s" % path, self.args.verbose)print_log("")