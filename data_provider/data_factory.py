import sys
import os
import numpy as np
parent_dir = os.path.dirname(os.getcwd())
current_dir = os.getcwd()
sys.path.append(parent_dir)
sys.path.append(current_dir)

from utils.preprocessing import split_by_season
from data_provider.data_loader import import_SUNCAM
from utils.nans import deal_with_nan 

#from sklearn.decomposition import PCA
#from data_provider.data_loader import 


def data_provider(_loc_data,include_time,csi,exclude_nights,season, nora, cam, data_encoder,deduct_cs, order = "manual"):
    
    if order == "save":
        flat_train, target_train, flat_valid,target_valid, flat_test,target_test = data_manual_load(_loc_data,include_time,csi,exclude_nights,season,nora, cam, data_encoder,deduct_cs)
        data_save(flat_train, target_train, flat_valid,target_valid, flat_test,target_test,include_time,csi,exclude_nights,season, nora, cam, data_encoder,deduct_cs)
        print("Data_Provider: Saved Data...")
    elif order == "load":
        try:
            flat_train, target_train, flat_valid,target_valid, flat_test,target_test = data_load(include_time,csi,exclude_nights,season, nora, cam, data_encoder,deduct_cs)
            print("Data_Provider: Are you absolute sure that this will be the right dataset?")
        except:
            print("Data_Provider: Loading failed. Going Manual...")
            flat_train, target_train, flat_valid,target_valid, flat_test, target_test = data_manual_load(_loc_data,include_time,csi,exclude_nights,season,nora, cam, data_encoder,deduct_cs)
    elif order == "manual":
        flat_train, target_train, flat_valid,target_valid, flat_test, target_test = data_manual_load(_loc_data,include_time,csi,exclude_nights,season,nora, cam, data_encoder,deduct_cs)
        print("Data_Provider: Performed a data reload...")
    else:
        raise KeyError
    return flat_train, target_train, flat_valid,target_valid, flat_test,target_test
    

def data_manual_load(_loc_data,include_time,csi,exclude_nights,season, nora, cam, data_encoder,deduct_cs):
    if include_time:
        time_for_plotting,et,src = import_SUNCAM(CSI = csi,Time = include_time,path = _loc_data+"/data/",
                                         exclude_nights = exclude_nights, include_cam = cam,deduct_cs = deduct_cs)
    else:
        et,src = import_SUNCAM(CSI = csi,Time = include_time,path = _loc_data+"/data/",
                                            exclude_nights = exclude_nights, include_cam = cam,deduct_cs=deduct_cs)
    if nora:
        nora_dat = np.load(_loc_data+"/data/nora3_CONCAT_20latlons.npy")
        src = np.concatenate((src,nora_dat[:,:,6:]),axis=0)
    src = src.astype('float32')
    # nan madness
    nanmask = np.isnan(src)
    src_nonan = deal_with_nan(src,nanmask)

    if season != "all": # look only at winter or summer depending on settings
        src_nonan,et,time_for_plotting = split_by_season(data =src_nonan,time = time_for_plotting, embedded_time=et,seasons=season)

    # do splits
    #split = (2,1,1,4) # meaning we split the data in 4. 2 get training, 1 validation, 1 testing
    # This is all old. So we do 5 years, meaning we want the first year as test, the second as valid and the rest 3 years as training
    i = int((1/5)*src.shape[2])
    j = int((2/5)*src.shape[2])
    train = src_nonan[:,:,j:]
    valid = src_nonan[:,:,i:j]
    test = src_nonan[:,:,:i]

    target_train = train[0,11,:]
    target_valid = valid[0,11,:]
    target_test = test[0,11,:]
    if data_encoder == "None":
        flat_train = flatten(train)
        flat_valid = flatten(valid)
        flat_test = flatten(test)
    else:
        flat_train,flat_test,flat_valid = load_encoded_data(data_encoder,i=i,j=j)
    
    if include_time:
        time_for_plotting = time_for_plotting[j:]


    # add embedded time to the flatted training sets
    if include_time:
        flat_train = np.concatenate((flat_train,et[j:,:]),axis = 1)
        flat_valid = np.concatenate((flat_valid,et[i:j,:]),axis = 1)
        flat_test = np.concatenate((flat_test,et[:i,:]),axis = 1)

    return flat_train, target_train, flat_valid,target_valid, flat_test,target_test

def load_encoded_data(encoder,i,j):
    if encoder == "autoencoder": 
        transformed = np.load("data/autoencoded_data.npz", allow_pickle=True)
        transformed_data = transformed["data"]
        flat_train = transformed_data[j:,:]
        flat_valid = transformed_data[i:j,:]
        flat_test = transformed_data[:i,:]

    elif encoder == "pca":
        transformed = np.load("data/pcaed_data.npz", allow_pickle=True)
        transformed_data = transformed["data"]
        flat_train = transformed_data[j:,:]
        flat_valid = transformed_data[i:j,:]
        flat_test = transformed_data[:i,:]
    else:
        raise KeyError
    return flat_train,flat_test,flat_valid

def construct_filename(time, csi,exclude_nights,season, nora, cam, deduct_cs):
    if time:
        ntime = "_time"
    else:
        ntime = "_notime"
    if csi:
        ncsi = "_csi"
    else:
        ncsi = "_nocsi"
    if exclude_nights:
        nnight = "_nonights"
    else:
        nnight = "_nights"
    season = "_" +season+"season"
    if nora:
        nnora = "_nora"
    else:
        nnora = "_nonora"
    if cam:
        ncam = "_cam"
    else:
        ncam = "_nocam"
    if deduct_cs:
        decs = deduct_cs
    else:
        decs = "nodecs"
    filename = "readyDat" + ntime+ ncsi + nnight + season + nnora + ncam + decs
    return filename

def data_save(flat_train, target_train, flat_valid,target_valid, flat_test,target_test,time,csi,exclude_nights,season, nora, cam,data_encoder,deduct_cs):
    filename = construct_filename(time,csi,exclude_nights,season, nora, cam,deduct_cs)
    np.savez("data/"+filename + ".npz",
             flat_train=flat_train,
             target_train = target_train,

             flat_valid=flat_valid,
             target_valid = target_valid,

             flat_test=flat_test,
             target_test = target_test
             )
    
def data_load(time,csi,exclude_nights,season, nora, cam,data_encoder,deduct_cs):
    filename = construct_filename(time,csi,exclude_nights,season, nora, cam,deduct_cs)
    data = np.load("data/"+filename+ ".npz")
    if data_encoder != "None": 
        print("Warning: Encoded data may not be compatible with selected Datasets.")
        raise TypeError
    print("Data Provider: Data Loaded. Filename = "+ filename)
    return data['flat_train'], data['target_train'], data['flat_valid'] ,data['target_valid'], data['flat_test'], data['target_test']

def flatten(data):
    flattened = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
    flattened = np.transpose(flattened, (1, 0))
    return flattened


def data_load_raw(_loc_data):
    """
    Raw data load except that NANs are dealt with as per nan policy.
    (Simple remove for now)
    """
    time_for_plotting,et,src = import_SUNCAM(CSI = True,Time = True,path = _loc_data+"/data/",
                                         exclude_nights = False, include_cam = True)

    nora_dat = np.load(_loc_data+"/data/nora3_CONCAT_20latlons.npy")
    src = np.concatenate((src,nora_dat[:,:,6:]),axis=0)
    src = src.astype('float32')
    # nan madness
    nanmask = np.isnan(src)
    src_nonan = deal_with_nan(src,nanmask)
    return src_nonan, et, time_for_plotting