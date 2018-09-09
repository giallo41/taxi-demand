
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import keras
import keras.backend as K

ADJ_RATE = 100

############################################################################
### Data Processing 
############################################################################

def load_np_data(file_name = '../datasets/seo_2017_7_12.npz'):
    
    
    try:
        load_datasets2=np.load(file_name)
        data_array = np.zeros(load_datasets2['arr_0'].shape)
        data_array = load_datasets2['arr_0']
        print (file_name, 'Loaded')
    except:
        print ('File Load Error! ')
        return -1
        
    return data_array

def logscale(data):
    return np.log(data+1)

def inverse_logscale(data):
    rtn = np.exp(data)-1
    rtn = rtn.astype(int)
    return rtn

def maxscale(data) : 
    return (data / 5000.0) * 10

def inverse_maxscale(data):
    rtn = (data*ADJ_RATE)
    rtn = rtn.astype(int)
    return rtn

def make_maxscale_data(data):
    rtn = inverse_logscale(data)
    return maxscale(rtn)


## Make One-hot-Temporal Data
## Week -> 50x50x7 
## Time -> 50x50x48
## holi -> 50x50x2 

def make_one_hot_data(data, row=50, col=50):
    
    m=data.shape[0]
    max_ch = int(data.max())+1
    
    rtn_data = []
    
    one_layer = np.ones((row,col,1))
    
    for i in range(m):
        
        
        idx = data[i,0]
        tmp_data = np.zeros((row,col,max_ch))
        tmp_data[:,:,idx:idx+1] = one_layer
        
        rtn_data.append(tmp_data)
        
    return np.asarray(rtn_data)

def make_binary_one_hot(data, row=50, col=50):
    m=data.shape[0]
    max_ch = 1
    rtn_data = []
    
    for i in range(m):
            
        value = data[i,0]
        
        if value == 0 :
            one_layer = np.zeros((row,col,max_ch))
        else:
            one_layer = np.ones((row,col,max_ch))
            
        rtn_data.append(one_layer)
        
    return np.asarray(rtn_data)
    
############################################################################


############################################################################
### Metric
############################################################################

def rmse(y_true,y_pred):
    
    
    rtn = np.sqrt(  np.average( np.square(y_pred-y_true) ) )
    
    return  rtn 

def mape(y_true,y_pred):
    
    rtn = np.mean(np.abs((y_true - y_pred) / (1.0+y_true)))
    
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

#######################################################################


#######################################################################
## Keras Fit Fuction
#######################################################################

def invlog_mape_tr10(y_true,y_pred):
    
    y_true = K.exp(y_true)-1
    y_pred = K.exp(y_pred)-1

#    true_mask = y_true>10
#    true_mask = K.cast_to_floatx(true_mask)
    true_mask = K.greater(y_true,10)
    true_mask = K.cast(true_mask, dtype=K.floatx())
    
    return K.mean(K.abs(  ( tf.boolean_mask ( (y_true - y_pred), true_mask )  / tf.boolean_mask(y_true, true_mask) ) ) )
                  
def invlog_rmse_tr10(y_true,y_pred):
    
    y_true = K.exp(y_true)-1
    y_pred = K.exp(y_pred)-1
    
#    true_mask = y_true>10
    true_mask = K.greater(y_true,10)
    true_mask = K.cast(true_mask, dtype=K.floatx())

    return K.sqrt( K.mean(K.square( tf.boolean_mask ((y_true - y_pred), true_mask ) ) ) )

def inv_minmax_mape_tr10(y_true,y_pred):
    
    y_true = y_true*ADJ_RATE
    y_pred = y_pred*ADJ_RATE

#    true_mask = y_true>10
#    true_mask = K.cast_to_floatx(true_mask)
    true_mask = K.greater(y_true,10)
    true_mask = K.cast(true_mask, dtype=K.floatx())
    
    return K.mean(K.abs(  ( tf.boolean_mask ( (y_true - y_pred), true_mask )  / tf.boolean_mask(y_true, true_mask) ) ) )

def inv_minmax_rmse_tr10(y_true,y_pred):
    
    y_true = y_true*ADJ_RATE
    y_pred = y_pred*ADJ_RATE
    
#    true_mask = y_true>10
    true_mask = K.greater(y_true,10)
    true_mask = K.cast(true_mask, dtype=K.floatx())

    return K.sqrt( K.mean(K.square( tf.boolean_mask ((y_true - y_pred), true_mask ) ) ) )


def invlog_mape(y_true,y_pred):
    
    
    y_true = K.exp(y_true)-1
    y_pred = K.exp(y_pred)-1
#    rtn = rtn.astype(int)
    
    return K.mean(K.abs((y_true - y_pred) / (1.0+y_true)))

def invlog_rmse(y_true, y_pred):
    
    y_true = K.exp(y_true)-1
    y_pred = K.exp(y_pred)-1
    
    return K.sqrt( K.mean(K.square(y_true - y_pred) ) )

    
def mae_t1(y_true,y_pred):
    
    sub = tf.subtract(y_pred[:,:,:,:1], y_true[:,:,:,:1])
    return tf.reduce_mean(tf.abs(sub))

def mae_t2(y_true,y_pred):
    
    sub = tf.subtract(y_pred[:,:,:,-1:], y_true[:,:,:,-1:])
    return tf.reduce_mean(tf.abs(sub))



def invlog_mape_t1(y_true,y_pred):
    
    y_true = K.exp(y_true)-1
    y_pred = K.exp(y_pred)-1
    
    t1_pred = y_pred[:,:,:,:1]
    t1_true = y_true[:,:,:,:1]

    return K.mean(K.abs((t1_true - t1_pred) / (1.0+t1_true)))

def invlog_mae_t2(y_true,y_pred):
    
    y_true = K.exp(y_true)-1
    y_pred = K.exp(y_pred)-1
    
    t2_pred = y_pred[:,:,:,-1:]
    t2_true = y_true[:,:,:,-1:]

    return K.mean(K.abs((t2_true - t2_pred) / (1.0+t2_true)))


def invlog_rmse_t1(y_true,y_pred):
    
    y_true = K.exp(y_true)-1
    y_pred = K.exp(y_pred)-1
    
    t1_pred = y_pred[:,:,:,:1]
    t1_true = y_true[:,:,:,:1]
    
    return K.sqrt( K.mean(K.square(t1_true - t1_pred) ) )

def invlog_rmse_t2(y_true,y_pred):
    
    y_true = K.exp(y_true)-1
    y_pred = K.exp(y_pred)-1
    
    t2_pred = y_pred[:,:,:,-1:]
    t2_true = y_true[:,:,:,-1:]
    
    return K.sqrt( K.mean(K.square(t2_true - t2_pred) ) )




#######################################################################
## Output Check
#######################################################################

        
def event_metric(y_true, y_pred, time_avg=8, base_lag = 8):
    np_pred = y_pred
    np_true = y_true
    
    ## Key - Col , Item - Index List 
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
    print ('')
    print ('## ---- Event Metric -----')
#    print (np.shape(event_true), np.shape(event_pred) )
    print ('- True Max %0.0f'%np.max(event_true), ', Pred Max %.0f'%np.max(event_pred))
    print ('- True Avg %0.3f'%np.average(event_true), ', Pred Avg %.3f'%np.average(event_pred))
    print ('- Event MAPE : %.3f'%mape(event_true,event_pred))
    print ('- Event RMSE : %.3f'%rmse(event_true,event_pred))



def hotspot_metric(y_true, y_pred):
    np_pred = y_pred
    np_true = y_true
    
    ## Key - Spot , Item - Col List 
    event_data_dic = {
        'Jongro' : [1174, 1175, 1176, 1177, 1178, 1224, 1225, 1226, 1227, 1228, 1274, 1275, 1276, 1277, 1278],
        'Hongdae' : [1316, 1317, 1318, 1319, 1320, 1367, 1368, 1369, 1418, 1419, 1420, 1421, 1422],
        'Iteawon' : [1526, 1527, 1528, 1574, 1577],
        'Apgujung' : [1630, 1631, 1632, 1633, 1634, 1680, 1681, 1682, 1683, 1684 ],
        'Gangnam' : [1730, 1731, 1732, 1733, 1734, 1780, 1781, 1782, 1783, 1784 ], 
        'Songpa' : [ 1736, 1737, 1738, 1739, 1740, 2042,2043] ,
        'Kwanac' : [2073, 2074, 2075, 2120]
    }
    
    
    event_true = []
    event_pred = []

    print ('')
    print ('---- Hotspot Metric -----')
    
    for key in event_data_dic:
    #    print (key)
        key_spot = key 
        spot_true = []
        spot_pred = []
        
        for item in event_data_dic[key]:
    #        print (item)
            tmp_pred = list(np_pred[:,item])
            tmp_true = list(np_true[:,item])

            spot_true = spot_true + tmp_true
            spot_pred = spot_pred + tmp_pred
            
            event_true = event_true+tmp_true
            event_pred = event_pred+tmp_pred
        
        spot_true = np.asarray(spot_true)
        spot_pred = np.asarray(spot_pred)
#        print ('-', key_spot, ' Area -----')
#        print ('  - Avg-', 'True : %.3f'%np.average(spot_true), ', Pred : %.3f'%np.average(spot_pred))
#        print ('  -', key_spot, 'MAPE : %.3f'%mape(spot_true,spot_pred))
#        print ('  -', key_spot, 'RMSE : %.3f'%rmse(spot_true,spot_pred))

    event_true = np.asarray(event_true)
    event_pred = np.asarray(event_pred)
    print ('')
    print ('## ---- Spot Total Metric -----')
    print ('- Avg-', 'True : %.3f'%np.average(event_true), ', Pred : %.3f'%np.average(event_pred))
    print ('- Spot Total MAPE : %.3f'%mape(event_true,event_pred))
    print ('- Spot Total RMSE : %.3f'%rmse(event_true,event_pred))



def model_metric(file_name, time_avg=4, base_lag = 8):

    df_true = pd.read_csv(file_name+'_gt.csv').drop('Unnamed: 0', axis=1)
    df_pred = pd.read_csv(file_name+'_pred.csv').drop('Unnamed: 0', axis=1)
#    df_loss = pd.read_csv(file_name+'_loss_metric.csv').drop('Unnamed: 0', axis=1)
    
    print ('## Model Name : ', file_name)
#    print ('- Val_Loss, log-MAE : %.3f'%df_loss.iloc[-1,1])
    print ('')
    print ('## ---- Base Metric -----')
    print ('- True Max %0.0f'%df_true.values.max(), ', Pred Max %.0f'%df_pred.values.max())
    print ('- True Avg %0.2f'%np.average(df_true.values), ', Pred Avg %.2f'%np.average(df_pred.values))
    print ('- MAPE(+1) : %.3f'%mape(df_true.values,df_pred.values))
    print ('- RMSE : %.3f'%rmse(df_true.values,df_pred.values))
    
    print ('')
    print ('## ---- Trs MAPE Metric -----')
    mape_list = []
    for trs in range(1, 16):
        
        tmp_trs_mape = mape_trs(df_true.values,df_pred.values, trs=trs)
        
        mape_list.append(tmp_trs_mape)
        
        print ('- %.0f or More'%trs, ' MAPE : %.3f'%tmp_trs_mape)
        
    
    print ('')
    print ('## ---- Trs MAA Metric -----')
    
    maa_list = []
    trs_list = list(np.arange(0, 1.05, 0.1))
    for trs in trs_list:
        
        tmp_maa = maa_trs(df_true.values,df_pred.values, trs=trs)
        maa_list.append(tmp_maa)
        
        print ('- Range %.0f'%(trs*100),'%', ' MAA : %.2f'%(tmp_maa*100),'%')

    if df_true.values.shape[1] > 2490 :
    
        event_metric(df_true.values,df_pred.values, time_avg=4, base_lag = 8)
        hotspot_metric(df_true.values, df_pred.values)


def history_save(loss, metric, model_name='baseline', output_folder='../output_file/'):
    
    df_tmp = pd.DataFrame()
    df_tmp['val_loss'] = loss
    df_tmp['val_metric'] = metric
    
    last_metric = metric[-1]
    
    out_file_name = output_folder+model_name+'_loss_metric_invmape_%.3f'%(last_metric)+'.csv'
    df_tmp.to_csv(out_file_name)
    
    print (out_file_name, ' Val loss Saved')
    
def make_test_ouput(model, input_data, y_test, model_name='baseline', norm='log', output_folder='../output_file/'):

    pred_out = model.predict(input_data)
    if norm == 'log' :
        x_test_pred_inversed = inverse_logscale(pred_out)
        y_test_inversed = inverse_logscale(y_test)
    else:
        x_test_pred_inversed = pred_out*ADJ_RATE
        y_test_inversed = y_test*ADJ_RATE
    
    x_test_pred_inversed = x_test_pred_inversed.astype(int)
    y_test_inversed = y_test_inversed.astype(int)
        
        
    np_df = np.reshape(x_test_pred_inversed, 
                       (x_test_pred_inversed.shape[0],  x_test_pred_inversed.shape[1],x_test_pred_inversed.shape[2]))
    np_df = np.reshape(np_df, 
                       (x_test_pred_inversed.shape[0],  x_test_pred_inversed.shape[1]*x_test_pred_inversed.shape[2]))
    
    np_y_test = np.reshape(y_test_inversed, 
                           (y_test_inversed.shape[0],  y_test_inversed.shape[1],y_test_inversed.shape[2]))
    np_y_test = np.reshape(np_y_test, 
                           (y_test_inversed.shape[0],  y_test_inversed.shape[1]*y_test_inversed.shape[2]))
    
    m, col = np_y_test.shape
    
    col_name = []
    for i in range(0, col):
        col_name.append('col_'+str(i) )
        
                       
    df_test = pd.DataFrame(np_y_test, columns=col_name, index=np.arange(0,np_y_test.shape[0]))
    df_test_pred = pd.DataFrame(np_df, columns=col_name, index=np.arange(0,np_df.shape[0]))
    
    y_test_file = output_folder+model_name+'_gt.csv'
    x_pred_file = output_folder+model_name+'_pred.csv'
    
    df_test.to_csv(y_test_file)
    df_test_pred.to_csv(x_pred_file)
    
#    print ('MAPE original Data(+1) : %.3f'%mape(y_test_inversed, x_test_pred_inversed) )
    
    print ('Test True saved : ', y_test_file)
    print ('Test Pred saved : ', x_pred_file)
    print ('')
    
#    model_metric(output_folder+model_name, time_avg=4, base_lag = 8)

    
    



def make_test_2ch_ouput(model, input_data, y_test, model_name='baseline', norm='log', output_folder='../output_file/'):

    pred_out = model.predict(input_data)
    if norm == 'log' :
        x_test_pred_inversed = inverse_logscale(pred_out)
        y_test_inversed = inverse_logscale(y_test)
    else:
        x_test_pred_inversed = pred_out*ADJ_RATE
        y_test_inversed = y_test*ADJ_RATE
    
    x_test_pred_inversed = x_test_pred_inversed.astype(int)
    y_test_inversed = y_test_inversed.astype(int)
    

    pred_t1 = x_test_pred_inversed[:,:,:,:1]
    pred_t2 = x_test_pred_inversed[:,:,:,-1:]
    
    true_t1 = y_test_inversed[:,:,:,:1]
    true_t2 = y_test_inversed[:,:,:,-1:]
    
    
    pred_t1 = np.reshape(pred_t1, 
                       (pred_t1.shape[0],  pred_t1.shape[1],pred_t1.shape[2]))
    pred_t1 = np.reshape(pred_t1, 
                       (pred_t1.shape[0],  pred_t1.shape[1]*pred_t1.shape[2]))

    pred_t2 = np.reshape(pred_t2, 
                       (pred_t2.shape[0],  pred_t2.shape[1],pred_t2.shape[2]))
    pred_t2 = np.reshape(pred_t2, 
                       (pred_t2.shape[0],  pred_t2.shape[1]*pred_t2.shape[2]))
    
    true_t1 = np.reshape(true_t1, 
                       (true_t1.shape[0],  true_t1.shape[1],true_t1.shape[2]))
    true_t1 = np.reshape(true_t1, 
                       (true_t1.shape[0],  true_t1.shape[1]*true_t1.shape[2]))

    true_t2 = np.reshape(true_t2, 
                       (true_t2.shape[0],  true_t2.shape[1],true_t2.shape[2]))
    true_t2 = np.reshape(true_t2, 
                       (true_t2.shape[0],  true_t2.shape[1]*true_t2.shape[2]))
    
    m, col = true_t1.shape
    
    col_name = []
    for i in range(0, col):
        col_name.append('col_'+str(i) )
    
    df_true_t1 = pd.DataFrame(true_t1, columns=col_name, index=np.arange(0,true_t1.shape[0]))
    df_true_t2 = pd.DataFrame(true_t2, columns=col_name, index=np.arange(0,true_t2.shape[0]))
    
    df_pred_t1 = pd.DataFrame(pred_t1, columns=col_name, index=np.arange(0,pred_t1.shape[0]))
    df_pred_t2 = pd.DataFrame(pred_t2, columns=col_name, index=np.arange(0,pred_t2.shape[0]))
    
    
    y_true_t1_file = output_folder+model_name+'t1_gt.csv'
    x_pred_t1_file = output_folder+model_name+'t1_pred.csv'
    
    y_true_t2_file = output_folder+model_name+'t2_gt.csv'
    x_pred_t2_file = output_folder+model_name+'t2_pred.csv'
    
    
    df_true_t1.to_csv(y_true_t1_file)
    df_true_t2.to_csv(y_true_t2_file)
    df_pred_t1.to_csv(x_pred_t1_file)
    df_pred_t2.to_csv(x_pred_t2_file)
    
    print ('Test True T1 saved : ', y_true_t1_file)
    print ('Test True T2 saved : ', y_true_t2_file)
    print ('Test Pred T1 saved : ', x_pred_t1_file)
    print ('Test Pred T2 saved : ', x_pred_t2_file)
    
    
    print ('## Model Name : ', model_name)
#    print ('- Val_Loss, log-MAE : %.3f'%df_loss.iloc[-1,1])
    print ('')
    print ('## (T1) ---- Base Metric -----')
    print ('- True Max %0.0f'%df_true_t1.values.max(), ', Pred Max %.0f'%df_pred_t1.values.max())
    print ('- True Avg %0.2f'%np.average(df_true_t1.values), ', Pred Avg %.2f'%np.average(df_pred_t1.values))
    print ('- MAPE(+1) : %.3f'%mape(df_true_t1.values,df_pred_t1.values))
    print ('- RMSE : %.3f'%rmse(df_true_t1.values,df_pred_t1.values))
    
    print ('')
    print ('## (T2) ---- Base Metric -----')
    print ('- True Max %0.0f'%df_true_t2.values.max(), ', Pred Max %.0f'%df_pred_t2.values.max())
    print ('- True Avg %0.2f'%np.average(df_true_t2.values), ', Pred Avg %.2f'%np.average(df_pred_t2.values))
    print ('- MAPE(+1) : %.3f'%mape(df_true_t2.values,df_pred_t2.values))
    print ('- RMSE : %.3f'%rmse(df_true_t2.values,df_pred_t2.values))

    
def make_test_ouput_norm(model, input_data, y_test, model_name='baseline', norm='log', output_folder='../output_file/'):

    pred_out = model.predict(input_data)
    if norm == 'log' :
        x_test_pred_inversed = inverse_logscale(pred_out)
        y_test_inversed = inverse_logscale(y_test)
    else:
        x_test_pred_inversed = pred_out*ADJ_RATE
        y_test_inversed = y_test*ADJ_RATE
    
    x_test_pred_inversed = x_test_pred_inversed.astype(int)
    y_test_inversed = y_test_inversed.astype(int)
        
    np_df = np.reshape(x_test_pred_inversed, 
                       (x_test_pred_inversed.shape[0],  x_test_pred_inversed.shape[1],x_test_pred_inversed.shape[2]))
    np_df = np.reshape(np_df, 
                       (x_test_pred_inversed.shape[0],  x_test_pred_inversed.shape[1]*x_test_pred_inversed.shape[2]))
    
    np_y_test = np.reshape(y_test_inversed, 
                           (y_test_inversed.shape[0],  y_test_inversed.shape[1],y_test_inversed.shape[2]))
    np_y_test = np.reshape(np_y_test, 
                           (y_test_inversed.shape[0],  y_test_inversed.shape[1]*y_test_inversed.shape[2]))
    
    m, col = np_y_test.shape
    
    col_name = []
    for i in range(0, col):
        col_name.append('col_'+str(i) )
        
                       
    df_test = pd.DataFrame(np_y_test, columns=col_name, index=np.arange(0,np_y_test.shape[0]))
    df_test_pred = pd.DataFrame(np_df, columns=col_name, index=np.arange(0,np_df.shape[0]))
    
    y_test_file = output_folder+model_name+'_gt.csv'
    x_pred_file = output_folder+model_name+'_pred.csv'
    
    df_test.to_csv(y_test_file)
    df_test_pred.to_csv(x_pred_file)
    
#    print ('MAPE original Data(+1) : %.3f'%mape(y_test_inversed, x_test_pred_inversed) )
    
    print ('Test True saved : ', y_test_file)
    print ('Test Pred saved : ', x_pred_file)
    print ('')
    
    print ("## Test datasets Performance")
    print ("- MAPE(11 or more) : %.4f"%mape_trs(y_test_inversed, x_test_pred_inversed, 11))
    print ("- RMSE(11 or more) : %.4f"%rmse_trs(y_test_inversed, x_test_pred_inversed, 11))
    print ('')