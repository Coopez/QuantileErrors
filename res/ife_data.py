import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import params


def preprocess_ife_data(params:dict):
    # Load data
    data = pd.read_pickle('IFE_dataset_model/irr_df_20210622_20240618.pkl')
    source = pd.read_pickle('IFE_dataset_model/trainval_df.pkl')
    
    result = process_solar_irradiance(data).shift(1)
    result = interpolate_new_columns(source, result)
    result = embedd_time(result)

    result['CSI'] = result['GHI'] / (result['GHI_clear'])
 
    
    pd.to_pickle(result,'IFE_dataset_model/trainval_df_preprocessed.pkl')
    print('Data preprocessed and saved to IFE_dataset_model/trainval_df_preprocessed.pkl')
    return 
# def summarize_target(data:pd.DataFrame, factor = 6):
#     target = data.iloc[:,0] # assuming that the target is the first column
#     target = target.rolling(window=factor, min_periods=1).mean()
#     return target

def import_ife_data(params:dict, dtype = "float32",base_path = None):
    # Load data
    standard_path = 'IFE_dataset_model/trainval_df_preprocessed.pkl'
    path = standard_path if base_path is None else os.path.join(base_path, standard_path)
    trainval_df = pd.read_pickle(path)
    if params["target"] == 'CSI':
        trainval_df = trainval_df.iloc[:,[-1,7, 0, 1, 2, 3, 4, 5, 6, 10, 11, 12,13,14,15,16,17,18,19,20,21,22]] # features to use - 8 and 9 are out
    elif params['target'] == 'GHI':
        trainval_df = trainval_df.iloc[:,[7, -1,0, 1, 2, 3, 4, 5, 6, 10, 11, 12,13,14,15,16,17,18,19,20,21,22]]
    elif params['target'] == 'ERLING_SETTINGS':
        trainval_df = trainval_df.iloc[:,[7, 0, 1, 2, 3, 4, 5, 6, 10, 11, 12]]
    else:
        raise NotImplementedError
    index_trainval = pd.read_pickle('IFE_dataset_model/trainval_C_index.pkl')
    # target = summarize_target(trainval_df,params["target_summary"])
    # Split data
    if dtype == "float32":
        trainval_df = trainval_df.astype(np.float32)
    split = 0.78 #0.78 0.74  TODO find out why 0.78 returns the wrong day.
    train_index = index_trainval[:int(len(index_trainval)*split)]
    val_index = index_trainval[int(len(index_trainval)*(split)):]
    # train_data = trainval_df.iloc[:int(len(index_trainval)*split)]
    # val_data = trainval_df.iloc[int(len(index_trainval)*(split)):]
    train_data = trainval_df.loc[train_index]
    val_data = trainval_df.loc[val_index]

    # idx = list(range(len(trainval_df.index)))

    # train_idx = idx[:int(len(index_trainval)*split)]
    # val_idx = idx[int(len(index_trainval)*(split)):]
    # train_idx = train_data.index.tolist()  # Indices of train_data
    # val_idx = val_data.index.tolist()  
        # Indices of val_data
    if params["target"] == 'CSI':
        train_target = train_data['CSI']
        val_target = val_data['CSI']
    elif params['target'] == 'GHI' or params['target'] == 'ERLING_SETTINGS':
        train_target = train_data['GHI']
        val_target = val_data['GHI']
    cs_train = train_data["GHI_clear"]
    cs_val = val_data["GHI_clear"]
    return train_data, train_target, val_data, val_target, cs_train, cs_val, index_trainval,train_index.index, val_index.index 



#test,t,a,_ = import_ife_data(params)

def embedd_time(data:pd.DataFrame):
    data['hour_sin'] = np.sin(data.index.hour / 24 * 2 * np.pi)
    data['day_sin'] =  np.sin(data.index.day / 31 * 2 * np.pi)
    data['month_sin'] = np.sin(data.index.month / 12 * 2 * np.pi)

    data['hour_cos'] = np.cos(data.index.hour / 24 * 2 * np.pi)
    data['day_cos'] = np.cos(data.index.day / 31 * 2 * np.pi)
    data['month_cos'] = np.cos(data.index.month / 12 * 2 * np.pi)

    return data

def process_solar_irradiance(df, timestamp_col='date_time'):
    """

    """
    # Create a copy of the dataframe to avoid modifying the original
    processed_df = df.copy()
    
    # Columns for sensors
    sensor_columns = ['KZPR_IRHoSunP', 'solpark_albedo_downwell_irr', 'MIS_IRHoSunP']
    
    def fill_nans(group):
        # First, try to fill NaNs from other sensors
        for col in sensor_columns:
            group[col] = group.fillna(group[sensor_columns].drop(columns=[col]).mean())[col]
        
        return group
    def interpolate_nans_and_calculate_variance(group, window_hours=24, window_days=30):
        group = group.interpolate("time") # basically a linear interpolation as time steps are regular
        
        # Calculate variances for 24-hour and 30-day windows
        # Use the mean of sensor columns
        #mean_sensor_col = 'mean_sensor_irradiance'
        #group[mean_sensor_col] = group[sensor_columns].mean(axis=1)
        
        # 24-hour variance
        last_24h_data = group.rolling(window=f'{window_hours}h', min_periods=1).agg(['var', 'mean']) 
        group['last_24h_variance'] = last_24h_data["KZPR_IRHoSunP"]["var"]
        group['last_24h_mean'] = last_24h_data["KZPR_IRHoSunP"]['mean']
        
        # 30-day variance 
        last_30d_data = group.rolling(window=f'{window_days}D', min_periods=1).agg(['var', 'mean']) 
        group['last_30d_variance'] = last_30d_data["KZPR_IRHoSunP"]['var']
        group['last_30d_mean'] = last_30d_data["KZPR_IRHoSunP"]['mean']
        

        return group
    
    # Set timestamp as index if not already
    if timestamp_col not in processed_df.index.names:
        processed_df = processed_df.set_index(timestamp_col)
    
    # Sort index to ensure correct rolling window calculations
    processed_df = processed_df.sort_index()
    
    # Apply processing
    processed_df = processed_df.apply(fill_nans, axis = 1)
    processed_df = processed_df.drop(columns=['solpark_albedo_downwell_irr', 'MIS_IRHoSunP'])
    processed_df = interpolate_nans_and_calculate_variance(processed_df)
    
    return processed_df

def interpolate_new_columns(main_df, variance_df):
    """
    """
        # Ensure timezone consistency
    if main_df.index.tz is not None:
        if variance_df.index.tz is None:
        # Use 'infer' for ambiguous times
            variance_df.index = variance_df.index.tz_localize(main_df.index.tz, ambiguous=True,nonexistent='shift_forward')
        else:
            variance_df.index = variance_df.index.tz_convert(main_df.index.tz)
    # main_df = main_df.tz_localize(None) if main_df.index.tz is not None else main_df
    # variance_df = variance_df.tz_localize(None) if variance_df.index.tz is not None else variance_df
    
    # Select only variance-related columns
    variance_cols = [col for col in variance_df.columns if 'variance' in col or 'mean' in col]
    
    # Interpolation function using numpy
    def interpolate_column(column):
        return np.interp(
            main_df.index.astype(np.int64), 
            variance_df.index.astype(np.int64), 
            column.values
        )
    
    # Interpolate each variance column
    interpolated_data = {}
    for col in variance_cols:
        interpolated_data[col] = interpolate_column(variance_df[col])
    
    # Create a DataFrame with interpolated values
    interpolated_df = pd.DataFrame(
        interpolated_data, 
        index=main_df.index
    )
    
    # Merge with main dataset
    result_df = pd.concat([main_df, interpolated_df], axis=1)
    
    return result_df


if __name__ == "__main__":
    preprocess_ife_data(params)