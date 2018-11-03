from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout,ZeroPadding2D,BatchNormalization,Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers.convolutional import Cropping2D
from keras.layers.merge import Concatenate
from keras import backend as K
from keras.preprocessing import image
import scipy.io as sio
import h5py
from keras.models import Sequential, Model
import tensorflow as tf
from keras.optimizers import SGD
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
# model=load_model('my_model.h5');

from keras.backend.common import _EPSILON

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.40
set_session(tf.Session(config=config))

smooth = 0.0001
   
def dice_coef(y_true, y_pred):
    y_true_f = 1 - K.flatten(y_true)
    y_pred_f = 1 - K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def error(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
    
    
def acc(y_true, y_pred):
    y_true=K.flatten(y_true)
    y_pred=K.flatten(y_pred)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1) 
    
def get_unet_train():
    img_rows = 256
    img_cols = 256
    inputs = Input((img_rows, img_cols,3))
    conv1 = Convolution2D(32, 7, 7,name='conv1',kernel_initializer='glorot_uniform',border_mode='same')(inputs)
    conv1= BatchNormalization()(conv1)
    conv1= Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Convolution2D(32, 5, 5,name='conv2',kernel_initializer='glorot_uniform',border_mode='same')(pool1)
    conv2= BatchNormalization()(conv2)
    conv2= Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Convolution2D(64, 3, 3,name='conv3',kernel_initializer='glorot_uniform',border_mode='same')(pool2)
    conv3= BatchNormalization()(conv3)
    conv3= Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Convolution2D(128, 3, 3, name='conv4',kernel_initializer='glorot_uniform',border_mode='same')(pool3)
    conv4= BatchNormalization()(conv4)
    conv4= Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 =  Convolution2D(128, 3, 3,name='conv5',kernel_initializer='glorot_uniform',border_mode='same')(pool4)
    conv5= BatchNormalization()(conv5)
    conv5= Activation('relu')(conv5)
    US4   =  UpSampling2D(size=(2, 2))(conv5)
    UP4   = Concatenate(axis=-1)([US4,conv4])
    UP4 =  Convolution2D(128, 3, 3 ,name='conv5_2',kernel_initializer='glorot_uniform',border_mode='same')(UP4)
    UP4= BatchNormalization()(UP4)
    UP4= Activation('relu')(UP4)
    US3   =  UpSampling2D(size=(2, 2))(UP4)
    UP3   = Concatenate(axis=-1)([US3,conv3])
    UP3 =  Convolution2D(64, 3, 3, name='conv4_2',kernel_initializer='glorot_uniform',border_mode='same')(UP3)
    UP3= BatchNormalization()(UP3)
    UP3= Activation('relu')(UP3)
    US2   =  UpSampling2D(size=(2, 2))(UP3)
    UP2   = Concatenate(axis=-1)([US2,conv2])
    UP2 =  Convolution2D(32, 3, 3, name='conv3_2',kernel_initializer='glorot_uniform',border_mode='same')(UP2)
    UP2= BatchNormalization()(UP2)
    UP2= Activation('relu')(UP2)
    US1   =  UpSampling2D(size=(2, 2))(UP2)
    UP1   = Concatenate(axis=-1)([US1,conv1])
    UP1 = Convolution2D(32, 3, 3,name='conv2_2',kernel_initializer='glorot_uniform',border_mode='same')(UP1)
    UP1= BatchNormalization()(UP1)
    UP1= Activation('relu')(UP1)
    final= Convolution2D(1, 1, 1, activation='sigmoid')(UP1)
    model = Model(input=inputs, output=final)
    return model
    
batch_size =1
nb_epoch = 60
f = h5py.File('/home/cvpr/Anabik/Sounak/PlaqueSegData.mat','r');
Img = f.get('images')
Mask = f.get('Mask')
Mask=np.asarray(Mask)
Img=np.asarray(Img)
Mask=np.expand_dims(Mask, axis=0)
Mask=np.swapaxes(Mask, 0, 3)
Img=np.swapaxes(Img, 0, 3)

split_ratio=.1 
suffledIndex=np.random.permutation(np.shape(Img)[0])
length_valid_data=np.int32(np.round(np.size(suffledIndex)*split_ratio))
x_valid=Img[suffledIndex[:length_valid_data],:,:,:]
y_valid=Mask[suffledIndex[:length_valid_data],:,:,:]
x_train = np.delete(Img, suffledIndex[:length_valid_data], axis=0)
y_train = np.delete(Mask, suffledIndex[:length_valid_data], axis=0)

x_train=np.float32(x_train)
x_valid=np.float32(x_valid)
y_train=np.uint8(y_train)
y_valid=np.uint8(y_valid)

model = get_unet_train();
model.summary()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='SGD', loss=error,metrics=[dice_coef])
history=model.fit(x_train, y_train, validation_data=[x_valid, y_valid],batch_size= batch_size, epochs=nb_epoch, verbose=1, shuffle=True )
history_Path = "/home/cvpr/Anabik/Sounak/History.mat"
sio.savemat(history_Path, mdict={'history': history.history});
model_weight_Path = "/home/cvpr/Anabik/Sounak/Model_Weight.mat"
model.save_weights(model_weight_Path)
