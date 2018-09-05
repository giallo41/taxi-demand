########################################
## Import Module
########################################


import pandas as pd
import numpy as np
import tensorflow as tf

import random as rn
from utils import *
import os
os.environ['PYTHONHASHSEED'] = '0'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

import keras
import keras.backend as K

tf.set_random_seed(RANDOM_SEED)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Flatten, Reshape, Dropout, Conv2DTranspose
from keras.layers import Concatenate, BatchNormalization, Add
from keras.models import Model, Sequential
from keras.layers import InputLayer
from keras.utils.training_utils import multi_gpu_model

###################
## Hyperparam  - 맞게 수정하거나 Arg로 받아 온다. (Data Folder / Model Name / Model Ver ) 추가 
###################

LRATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 3000
SCALE = 'log' # or 'max' or 'log'
NGPU = 2

## Folder / File Name
DATA_FOLDER = '/data/public/rw/prj-mobility/10.195.12.143/kakaobrain/taxi/datasets/NY_data/'

## Baseline Model Folder 
BASE_MODEL_NAME = 'MODEL_V2_NYC'
MODEL_VER = '0905_002_NY_V1_min_max_scale_lr_0_0001'


##################################################
OUTPUT_FOLDER = '../output_file/'
TF_FOLDER = '../tfgraph/'
MODEL_SAVE_FOLDER = '../saved_model/'

MODEL_NAME_ = BASE_MODEL_NAME+'_'+MODEL_VER
OUTPUT_FOLDER = OUTPUT_FOLDER+BASE_MODEL_NAME+'/'
TF_FOLDER = TF_FOLDER+BASE_MODEL_NAME+'/'+MODEL_VER
MODEL_SAVE_FOLDER = MODEL_SAVE_FOLDER+BASE_MODEL_NAME+'/'

print ('Model Name :',MODEL_NAME_)

if os.path.isdir(OUTPUT_FOLDER) == False:
    os.makedirs(OUTPUT_FOLDER)
    print ('Output Folder :',OUTPUT_FOLDER, 'created')
else :
    print ('Output Folder :',OUTPUT_FOLDER)

if os.path.isdir(TF_FOLDER) == False:
    os.makedirs(TF_FOLDER)
    print ('Tfgraph Folder :',TF_FOLDER, 'created')
else :
    print ('Tfgraph Folder :',TF_FOLDER)
    
if os.path.isdir(MODEL_SAVE_FOLDER) == False:
    os.makedirs(MODEL_SAVE_FOLDER)
    print ('Save Folder :',MODEL_SAVE_FOLDER, 'created')
else :
    print ('Save Folder :',MODEL_SAVE_FOLDER)
    
    
########################################
## Data Load
########################################

ADJ_RATE = 100

NYC_FOLDER = '/data/public/rw/prj-mobility/10.195.12.143/kakaobrain/taxi/datasets/NY_data/'

x_st_train = load_np_data(NYC_FOLDER+'train_st_x_2d.npz') / ADJ_RATE
x_ed_train = load_np_data(NYC_FOLDER+'train_end_x_2d.npz') / ADJ_RATE
y_train = load_np_data(NYC_FOLDER+'train_st_y_2d.npz') / ADJ_RATE
temporal_train = load_np_data(NYC_FOLDER+'temporal_train.npz')


x_st_test = load_np_data(NYC_FOLDER+'test_st_x_2d.npz') / ADJ_RATE
x_ed_test = load_np_data(NYC_FOLDER+'test_end_x_2d.npz') / ADJ_RATE
y_test = load_np_data(NYC_FOLDER+'test_st_y_2d.npz') / ADJ_RATE
temporal_test = load_np_data(NYC_FOLDER+'temporal_test.npz')

coord_train = load_np_data(NYC_FOLDER+'train_coord_x_2d.npz')/10 #args.coord
coord_test = load_np_data(NYC_FOLDER+'test_coord_x_2d.npz')/10 #args.coord

x_st_train.max(), x_ed_train.max(), y_train.max(), x_st_test.max(), x_ed_test.max(), y_test.max()


x_st_train = x_st_train[:-1]
x_ed_train = x_ed_train[:-1]


x_ed2_test = x_ed_test[1:]

x_st2_test = x_st_test[1:]

x_st_test = x_st_test[:-1]
x_ed_test = x_ed_test[:-1]

y_t2_train = y_train[1:]
y_t2_test = y_test[1:]

y_train = y_train[:-1]
y_test = y_test[:-1]

x_t2_temporal_train = temporal_train[1:]
x_t2_temporal_test = temporal_test[1:]

x_t1_temporal_train = temporal_train[:-1]
x_t1_temporal_test = temporal_test[:-1]

x_st_train = np.concatenate((x_st_train,coord_train[:-1]), axis=3)
x_st_test = np.concatenate((x_st_test,coord_test[:-1]), axis=3)

print (x_st_train.shape, x_ed_train.shape, y_train.shape, x_t1_temporal_train.shape,x_t2_temporal_train.shape, y_t2_train.shape)
print (x_st_test.shape, x_ed_test.shape, y_test.shape, x_t1_temporal_test.shape,x_t2_temporal_test.shape, y_t2_test.shape)


########################################
## Make Validation Data About 20% 
########################################

## Make Validation Data About 20% 
val_idx = 1524

print ('----- Before Split ----')
#print (x_st_train.shape, x_ed_train.shape, x_t1_temporal_train.shape, x_t0_temporal_train.shape, x_t2_temporal_train.shape)

x_st_val = x_st_train[val_idx:]
x_ed_val = x_ed_train[val_idx:]
y_val = y_train[val_idx:]
x_t1_temporal_val = x_t1_temporal_train[val_idx:]
#x_t0_temporal_val = x_t0_temporal_train[val_idx:]
x_t2_temporal_val = x_t2_temporal_train[val_idx:]


x_st_train = x_st_train[:val_idx]
x_ed_train = x_ed_train[:val_idx]
y_train = y_train[:val_idx]
x_t1_temporal_train = x_t1_temporal_train[:val_idx]
#x_t0_temporal_train = x_t0_temporal_train[:val_idx]
x_t2_temporal_train = x_t2_temporal_train[:val_idx]


print ('----- After Split ----')
print (x_st_train.shape, x_ed_train.shape, x_t1_temporal_train.shape, x_t2_temporal_train.shape)
print (x_st_val.shape, x_ed_val.shape, x_t1_temporal_val.shape, x_t2_temporal_val.shape)

########################################
## Make Model
########################################


K.clear_session()
print ('Session Reseted')

model_input_train_data = [x_st_train, x_ed_train, x_t1_temporal_train, x_t2_temporal_train]
model_input_val_data=[x_st_val, x_ed_val, x_t1_temporal_val, x_t2_temporal_val]
model_input_test_data = [x_st_test, x_ed_test, x_t1_temporal_test, x_t2_temporal_test]


def make_temporal_model(st_data, ed_data, tmp_data):

    im_h,im_w,im_c = st_data[0].shape

    start_input = Input(shape=(st_data[0].shape))
    end_input = Input(shape=(ed_data[0].shape))
    t1_temporal_input = Input(shape=(tmp_data[0].shape))
    t2_temporal_input = Input(shape=(tmp_data[0].shape))

    input_tensors = [start_input, end_input, t1_temporal_input, t2_temporal_input]

    coord_input = keras.layers.Lambda(lambda xin: xin[:,:,:,-2:] )(start_input)
    start_input = keras.layers.Lambda(lambda xin: xin[:,:,:,:-2] )(start_input)

    #Guassian Noise Augmentation for Training

    ## 2a or 2b 추가 
    #temp_conv2d_1st = Conv2D(16, kernel_size=(1,1), strides=(1,1))
    #temp_conv2d_2nd = Conv2D(16, kernel_size=(1,1), strides=(1,1))

    #net_t1_temp = temp_conv2d_1st(t1_temporal_input)
    #net_t1_temp = temp_conv2d_2nd(net_t1_temp)
    
    #net_t2_temp = temp_conv2d_1st(t2_temporal_input)
    #net_t2_temp = temp_conv2d_2nd(net_t2_temp)

    # 1a or 2a 버전

    net_t1_temp = Conv2D(16, kernel_size=(1,1), strides=(1,1))(t1_temporal_input)
    net_t1_temp = Conv2D(16, kernel_size=(1,1), strides=(1,1))(net_t1_temp)

    #net_t2_temp = Conv2D(16, kernel_size=(1,1), strides=(1,1))(t2_temporal_input)
    #net_t2_temp = Conv2D(16, kernel_size=(1,1), strides=(1,1))(net_t2_temp)

    net1 = layers.concatenate([start_input, net_t1_temp], axis=-1)
    net1 = Conv2D(64, kernel_size=(3,3), activation=None, padding='same')(net1)
    net1 = layers.Activation('relu')(net1)
    net1 = BatchNormalization()(net1)

    net11 = AveragePooling2D(pool_size=(2,2), strides=(2,2))(net1)

    net2 = Conv2D(128, kernel_size=(3,3), activation=None, padding='same')(net11)
    net2 = layers.Activation('relu')(net2)
    net2 = BatchNormalization()(net2)

    net3 = Conv2D(128, kernel_size=(3,3), activation=None, padding='same')(net2)
    net3 = layers.Activation('relu')(net3)
    net3 = BatchNormalization()(net3)
    net33 = layers.concatenate([net2, net3], axis=-1)

    net4 = Conv2D(128, kernel_size=(3,3), activation=None, padding='same')(net33)
    net4 = layers.Activation('relu')(net4)
    net4 = BatchNormalization()(net4)
    net44 = layers.concatenate([net2, net3, net4], axis=-1)
    #net4 = layers.Add()([net2, net4])

    net5 = Conv2DTranspose(128, kernel_size=(2,2), strides=(2,2), padding='same')(net44)
    net5 = layers.Activation('relu')(net5)
    net5 = BatchNormalization()(net5)
    net5 = Dropout(0.5)(net5)

    net5 = layers.concatenate([net5, net1], axis=-1)

    net6 = Conv2DTranspose(256, kernel_size=(3,3), strides=(1,1), padding='same')(net5)
    net6 = layers.Activation('relu')(net6)
    net6 = BatchNormalization()(net6)
    net6 = Dropout(0.5)(net6)

    net_end = Conv2D(8, kernel_size=(3,3), padding='same')(end_input)
    net_end = BatchNormalization()(net_end)
    net_end = Conv2D(8, kernel_size=(3,3), padding='same')(net_end)
    net_end = BatchNormalization()(net_end)
    
    

    net71 = layers.concatenate([net6, net_t1_temp, net_end, start_input, end_input, coord_input], axis=-1)
    net71 = Conv2DTranspose(256, kernel_size=(1,1), padding='same')(net71)
    net71 = layers.Activation('relu')(net71)
    net71 = BatchNormalization()(net71)

    #net72 = layers.concatenate([net6, start_input, net_end, end_input, net_t2_temp, coord_input], axis=-1)
    #net72 = Conv2DTranspose(256, kernel_size=(1,1), padding='same')(net72)
    #net72 = layers.Activation('relu')(net72)
    #net72 = BatchNormalization()(net72)

    t1_output = Conv2D(1, kernel_size=(1,1), padding='same')(net71)
    output = layers.Activation('relu')(t1_output)

    #t2_output = Conv2D(1, kernel_size=(1,1), padding='same')(net72)
    #t2_output = layers.Activation('relu')(t2_output)

    #output = layers.concatenate([t1_output, t2_output], axis=-1)

    model = Model(inputs=input_tensors, outputs=output)
    return model
    
#    y3 = keras.layers.GaussianNoise(0.5)(y3)

b_model = make_temporal_model(x_st_train,x_ed_train, x_t1_temporal_train)
model = multi_gpu_model(b_model, gpus=NGPU)
print (MODEL_NAME_, 'Model Created')
print (b_model.summary())

########################################
## Callback 
########################################


from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam

from keras.callbacks import Callback

class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr) # K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('LR: {:.6f}'.format(lr))

tb_hist = keras.callbacks.TensorBoard(log_dir=TF_FOLDER, histogram_freq=0, write_graph=True, write_images=True)
early_stopping = keras.callbacks.EarlyStopping(monitor='inv_minmax_mape_tr10', min_delta=0, patience=100, verbose=0, mode='min')


########################################
## Training  
########################################


import time
start_time = time.time() 


model.compile(loss=['mean_squared_error'], optimizer=Adam(lr=LRATE,  decay=0.01), metrics=['mean_absolute_error', inv_minmax_rmse_tr10, inv_minmax_mape_tr10])
history = model.fit(model_input_train_data , y_train,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  callbacks=[tb_hist,early_stopping, SGDLearningRateTracker()],
                  shuffle = True,
                  verbose=2
                  ,validation_data=(model_input_val_data, y_val)
                 )
    
val_loss = history.history['val_loss']
val_metric = history.history['val_mean_absolute_error']

val_loss2 = history.history['val_loss']
val_metric2 = history.history['val_mean_absolute_error']

end_time = time.time()
n_epochs = len(val_metric2) #len(val_metric)+len(val_metric2)

print ('')
print("--- Train Time : %0.2f hour  ---" %(  (end_time - start_time)/3600  ))
print("--- # of Epochs: %0.f  ---" %( n_epochs ) )

print ('')
print ("## Test datasets Performance")
print ("- MAPE(11 or more) : %.3f"%mape_trs(y_test*ADJ_RATE, pred_test*ADJ_RATE, 11))
print ("- RMSE(11 or more) : %.3f"%rmse_trs(y_test*ADJ_RATE, pred_test*ADJ_RATE, 11))
print ('')

