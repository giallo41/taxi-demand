
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import pyplot


def logscale(data):
    return np.log(data+1)

def inverse_logscale(data):
    rtn = np.exp(data)-1
    rtn = rtn.astype(int)
    return rtn


############################################################################
### Metric
############################################################################

def rmse(y_true,y_pred):
    
    
    rtn = np.sqrt(  np.average( np.square(y_pred-y_true) ) )
    
    return  rtn 

def mape(y_true,y_pred):
    
    rtn = np.mean(np.abs((y_true - y_pred) / (1.0+y_true)))
    
    return rtn


def mae(y_true,y_pred):
    
    rtn = np.mean(np.abs((y_true - y_pred)) )
    
    return rtn
                  

#######################################################################
### Treshhold Metric 
#######################################################################

def rmse_trs(y_true,y_pred, trs=1):
    
    true_mask = y_true>=trs
    tmp_abs = np.square( (y_true-y_pred)[true_mask] )

    rtn = np.sqrt(np.average(tmp_abs))
    
    return rtn 

def mape_trs(y_true,y_pred, trs=1):

    true_mask = y_true>=trs
    tmp_abs = np.divide(np.abs(y_true-y_pred)[true_mask] , y_true[true_mask])

    rtn = (np.average(tmp_abs))
    
    return rtn 

def maa_trs(y_true,y_pred, trs=0.0):
    
    if trs == 1.0 :
        return 1
    else :
        true_mask = (y_pred>=y_true*(1.0-trs))&(y_pred<=y_true*(1.0+trs))
    
    return np.average(true_mask)


def mae_trs(y_true,y_pred, trs=1):
    
    true_mask = y_true>=trs
    rtn = np.mean(np.abs((y_true - y_pred)[true_mask]) )
    
    return rtn

#######################################################################

def chk_1ch_output(model_name):

    df_t1_pred = pd.read_csv(model_name+'_pred.csv')
    df_t1_true = pd.read_csv(model_name+'_gt.csv')
    
    try:
        df_t1_true = df_t1_true.drop('Unnamed: 0', axis=1)
        y_t1_true = df_t1_true.values
    except:
        y_t1_true = df_t1_true.values
        
    try:
        df_t1_pred = df_t1_pred.drop('Unnamed: 0', axis=1)
        y_t1_pred = df_t1_pred.values
    except:
        y_t1_pred = df_t1_pred.values
        

    #y_t1_pred = df_t1_pred.values
    #y_t1_true = df_t1_true.values
    
    y_t1_pred = y_t1_pred.astype(int)
    y_t1_true = y_t1_true.astype(int)
    
    y_t1_pred = np.expand_dims(y_t1_pred, axis=2)
    y_t1_true = np.expand_dims(y_t1_true, axis=2)
    
    
    print ('## Model Name :', model_name)
    print ('MAPE 11 : %.4f'%(mape_trs(y_t1_true, y_t1_pred, 11)))
    print ('RMSE 11 : %.4f'%(rmse_trs(y_t1_true, y_t1_pred, 11)))
    print ('')
    print ('MAPE all : %.4f'%(mape(y_t1_true, y_t1_pred)))
    print ('RMSE all : %.4f'%(rmse(y_t1_true, y_t1_pred)))
    print ('')
    print ('MAE 11 : %.4f'%(mae_trs(y_t1_true, y_t1_pred, 11)))
    print ('MAE all : %.4f'%(mae(y_t1_true, y_t1_pred)))
    print ('')
   
    
    
    mape_trs_list = []
    rmse_trs_list = []
    maa_trs_list = []
    
    trs_list = range(1, 21)
    
    for trs in trs_list:
        mape_trs_list.append(mape_trs(y_t1_true, y_t1_pred, trs))
        rmse_trs_list.append(rmse_trs(y_t1_true, y_t1_pred, trs))
    
    
    trs_list = list(np.arange(0, 1.05, 0.1))
    for trs in trs_list:
        
        tmp_maa = maa_trs(y_t1_true,y_t1_pred, trs=trs)
        maa_trs_list.append(tmp_maa)
        
    return mape_trs_list, rmse_trs_list, maa_trs_list

def chk_event_metric_3by3(model_name, avg_time=4, time_lag = 8):
    ## After Time_lag , Average data 
    ## 3x3 cell averaging 
    
    df_t1_pred = pd.read_csv(model_name+'_pred.csv')
    df_t1_true = pd.read_csv(model_name+'_gt.csv')
    
    try:
        df_t1_true = df_t1_true.drop('Unnamed: 0', axis=1)
        y_t1_true = df_t1_true.values
    except:
        y_t1_true = df_t1_true.values
        
    try:
        df_t1_pred = df_t1_pred.drop('Unnamed: 0', axis=1)
        y_t1_pred = df_t1_pred.values
    except:
        y_t1_pred = df_t1_pred.values
         
    y_t1_pred = y_t1_pred.astype(int)
    y_t1_true = y_t1_true.astype(int)
    
    np_pred = y_t1_pred
    np_true = y_t1_true
    
#    y_t1_pred = np.expand_dims(y_t1_pred, axis=2)
#    y_t1_true = np.expand_dims(y_t1_true, axis=2)
    
    event_data_dic = {
        
        1861 : [1511, 1556, 1604],     # 고척돔
        1214 : [546],# 상암동
        1327 : [1220, 1266], # 장충동
        1735 : [1220, 1266, 2520, 2567, 1846, 1892, 1940], # 잠실경기장
        
    }
    
    event_true = []
    event_pred = []

    for key in event_data_dic:
    #    print (key)
        for item in event_data_dic[key]:
            
            key_list = [key-51, key-50, key-49, key-1, key, key+1, key+49, key+50, key+51]
            
            for ky in key_list:
    #        print (item)
                tmp_pred = list(np_pred[item+time_lag:item+time_lag+avg_time, ky])
                tmp_true = list(np_true[item+time_lag:item+time_lag+avg_time, ky])

                event_true = event_true+tmp_true
                event_pred = event_pred+tmp_pred

    event_true = np.asarray(event_true)
    event_pred = np.asarray(event_pred)
#    print ('')
    print ('## ---- Event Metric -----')
#    print (np.shape(event_true), np.shape(event_pred) )
#    print ('## Model Name :',model_name)
    print ('- True Max %0.0f'%np.max(event_true), ', Pred Max %.0f'%np.max(event_pred))
    print ('- True Avg %0.4f'%np.average(event_true), ', Pred Avg %.4f'%np.average(event_pred))
    print ('- Event MAPE : %.4f'%mape(event_true,event_pred))
    print ('- Event RMSE : %.4f'%rmse(event_true,event_pred))
    print ('- Event MAE : %.4f'%mae(event_true,event_pred))

#######################################################################  
    
def plot_img(img):

    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = plt.subplot(111)  
    tmp_img = []
    
    data_max = img.max()
    
    img = 40*np.log(img+1)
    img = np.minimum(img, 255)
    
#    if  data_max < 100 : 

#        img = img / data_max
#        img = img*100
#    else :        
    
    img_r = 28*np.log(img+1) #np.minimum(img, 255) #img/data_max
#    img_r = img_r*255
    
    img_g = 28*np.log(img+1) #np.minimum(img, 255) #img/data_max
#    img_g = img_g*255
    
    img_b = 28*np.log(img+1) #np.minimum(img, 255) #img/data_max #np.zeros(img.shape)
#    img_b = img_b*255

    img_r = img_r.astype(int)
    img_g = img_g.astype(int)
    img_b = img_b.astype(int)
    img = img.astype(int)
#    img = img *200

#    print (img.max())

    if len(img.shape)<3 or img.shape[2] < 3 : 

        
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        img_r = np.reshape(img_r, (img_r.shape[0], img_r.shape[1], 1))
        img_g = np.reshape(img_g, (img_g.shape[0], img_g.shape[1], 1))
        img_b = np.reshape(img_b, (img_b.shape[0], img_b.shape[1], 1))
#        tmp_img = np.concatenate((img_r,img_g,img_b), axis=2)
        tmp_img = np.concatenate((img,img,img), axis=2)

    
    #plt.tick_params(axis=None, which="off", bottom="off", top="off",    
    #                labelbottom="off", left="off", right="off", labelleft="off") 
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
#    plt.figure(figsize=(8, 8))
    plt.imshow(tmp_img, cmap='gray')
    plt.show()
    return fig
