from tensorflow.keras.utils import Sequence

import numpy as np
import os
import skimage
from skimage.io import imread
from model_utils import parser



class DataGenerator(Sequence):
    '''
    Multi-threaded data generator
    
    Each thread reads a batch of images and their object labels
    Arguments:
        args:                   User-defined configuration
        dictionary:             Dictionary of image filenames and object labels
        n_classes(int):         Number of object classes
        feature_shapes(tensor): Shape of ssd head feature maps
        n_anchors(int):         Number of anchor boxes per feature map pt
        shuffle(Bool):          If dataset should be shuffled bef sampling

    
    '''
    def __init__(self,
                args,
                shuffle=True
                ) :
        self.args = args 
        self.input_shape = (args.height,args.width,args.channels)
        self.shuffle = shuffle 
        self.on_epoch_end()
        self.get_dictionary()

    
    def __len__(self):
        '''Number of batches per epoch'''
        blen = np.floor(len(self.dictionary)/self.args.batch_size)
        return int(blen)
    
    def __getitem__(self, index):
        '''Get a batch of data'''
        start_index = index *self.args.batch_size 
        end_index = (index+1)*self.args.batch_size 
        keys = self.keys[start_index:end_index]
        x,y = self.__data_generation(keys)
        return x,y 
    
    def get_dictionary(self):
        '''load ground truth dictionary of image filename:segmentation masks'''
        path = os.path.join(self.args.data_path,self.args.train_labels)
        self.dictionary = np.load(path,allow_pickle=True).flat[0]    
        labels = self.dictionary[self.keys[0]]
        self.n_classes = labels.shape[-1]
    

    def on_epoch_end(self):
        '''Shuffle after each epoch'''
        if self.shuffle==True:
            np.random.shuffle(self.keys)
    
    
    def __data_generation(self,keys):
        '''Generate train data: image and object detection ground truth labels
        Arguments:
            keys(array):    Randomly sampled keys (key is image filename)
        Returns:
            x(tensor):      Batch images
            y(tensor):      Batch classes,offsets and masks
            
        '''
        # a batch of images
        x = []
        # and their corresponding segmentation masks 
        y = []

        for i,key in enumerate(keys):
            # images are assumed to be stored 
            # in self.args.data_path 
            # key is the image filename 
            image_path = os.path.join(self.args.data_path,key)
            image = skimage.img_as_float(imread(image_path))
            # append image to the list 
            x.append(image)
            # and its corresponding label (segmentation mask)
            labels = self.dictionary[key]
            y.append(labels)
        
            
        return np.array(x),np.array(y)
      


if __name__ =="__main__":
    parser = parser()
    args = parser.parse_args()
    data_gen = DataGenerator(args)
    images, labels = data_gen.__getitem__(0)
    
    import matplotlib.pyplot as plt
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Input image', fontsize=14)
    plt.imshow(images[0])
    plt.savefig("input_image.png", bbox_inches='tight')
    plt.show()

    labels = labels * 255
    masks = labels[..., 1:]
    bgs = labels[..., 0]

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Semantic segmentation', fontsize=14)
    plt.imshow(masks[0])
    plt.savefig("segmentation.png", bbox_inches='tight')
    plt.show()

    shape = (bgs[0].shape[0], bgs[0].shape[1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Background', fontsize=14)
    plt.imshow(np.reshape(bgs[0], shape), cmap='gray', vmin=0, vmax=255)
    plt.savefig("background.png", bbox_inches='tight')
    plt.show()
    
