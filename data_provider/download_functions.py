import pandas as pd
import numpy as np

#import netCDF4
# import xarray as xr
# import pyproj
import multiprocessing
import time
import os
from datetime import timedelta

# import cdsapi # for CAN


class NORA3():
    def __init__(self,latlons: list, vars: list, start_date: str, end_date: str, save_range: int, sampling: bool) -> None:
        
        self.state = start_date # represents how far the download has come. May be used to restart in case it crashes/ needs to be stopped
        self.sampling = sampling # this samples only every 15th days of the month. This results in less data to download that can maybe still be used for ANOVA analysis

        self.latlons = latlons
        self.vars = vars
        self.start_date = start_date
        self.end_date = end_date
        self.save_range = save_range
        

    def download_nora3(self):
        # NOW LEGACY AND NOT PROPERLY DEBUGGED.
        print("Use download_nora3_parallel instead!")
        raise NotImplemented

        """
        The idea here is to download per year, month, day, 6hour cohort and per hour. It gets tricky, as in the 6 hour cohort there are all files one by one
        I want to concat every 6 hours ideally.
        Problem is that the download takes ages. Not sure if one can speed it up apart from trying paralellization though. 
        
        """
        timerange = pd.date_range(start=self.start_date, end = self.end_date, freq = 'H')
        

        upload_time = ['00','06','12','18']
        casting_set = ['04','05','06','07','08','09']
        shifted_range = range(4, 24)  
        wrapped_range = range(0,  4)
        upload_permutations = [[hour6,hour] for hour6 in upload_time for hour in casting_set]
        h_lookup = {str(hour).zfill(2): element for hour, element in zip(shifted_range,upload_permutations)}
        h_lookup.update({str(hour).zfill(2): element for hour, element in zip(wrapped_range,upload_permutations[-4:])})
        
        self.save_range = self.save_range +1 # due to the way its set up it needs one more to achieve the desired length
        save_counter = 1
        export = np.zeros((len(vars),len(self.latlons),self.save_range))*np.nan
        
        for i,hour in enumerate(timerange):

            hour_str = hour.strftime("%H")
            
            if int(hour_str)>3: ## need this as anything below will be a different day
                date = timerange.date[i]
            else:
                date = timerange.date[i] - pd.DateOffset(days=1)

            org_path = "https://thredds.met.no/thredds/dodsC/nora3/"
            path = org_path + date.strftime("%Y/%m/%d")+"/"+ h_lookup[hour_str][0] +"/fc"+ date.strftime("%Y%m%d") + h_lookup[hour_str][0] +"_0" + h_lookup[hour_str][1] + "_fp.nc"
            #print("Path = " + path+ " ///////// Date = " + str(timerange[i]) )
            DS = xr.open_dataset(path)
            x = DS.variables["x"].data
            y = DS.variables["y"].data
            proj = pyproj.Proj(DS.variables["projection_lambert"].attrs["proj4"]) # get projection
            A  = np.zeros((len(vars),len(self.latlons))) * np.nan

            for varidx, var in enumerate(vars): 
                a = DS.variables[var].data[0,0,:,:]# results from format (1,1,x,y)
                for ptidx, latlon in enumerate(self.latlons):
                    lat = latlon[0]
                    lon = latlon[1]
                    # Compute projected coordinates of lat/lon point
                    X,Y = proj(lon,lat)
                    Ix = np.argmin(np.abs(x - X)) # take location which is closest
                    Iy = np.argmin(np.abs(y - Y)) 
                    A[varidx,ptidx] = a[Iy,Ix] 
                
            if save_counter % self.save_range == 0: # wanna save every X hours
                filename = "nora3_{0}_{2}hours_{1}vars_{3}latlons.npy".format(timerange[i].strftime("%Y%m%d_%H"),len(vars),self.save_range-1,len(self.latlons))
                np.save(filename, export)
                print(filename)
                export = np.zeros((len(vars),len(self.latlons),self.save_range))* np.nan
            else:
                export[:,:,save_counter % self.save_range -1] = A
                #print("me alive")
            save_counter +=1



    def process_hour(self, hour, h_lookup):
        hour_str = hour.strftime("%H")
        self.state = hour.strftime("%Y-%m-%d-%H")
        if int(hour_str) > 3:
            date = hour.date()
        else:
            date = hour.date() - pd.DateOffset(days=1)

        org_path = "https://thredds.met.no/thredds/dodsC/nora3/"
        #addon may be able to push the whole value extraction to the server, thus speed up downloadtime
        addon = "?projection_lambert,x[0:1:888],y[0:1:1488],integral_of_surface_net_downward_shortwave_flux_wrt_time[0:1:0][0:1:0][0:1:1488][0:1:888],air_temperature_2m[0:1:0][0:1:0][0:1:1488][0:1:888],relative_humidity_2m[0:1:0][0:1:0][0:1:1488][0:1:888],cloud_area_fraction[0:1:0][0:1:0][0:1:1488][0:1:888],surface_air_pressure[0:1:0][0:1:0][0:1:1488][0:1:888],wind_direction[0:1:0][0:1:0][0:1:1488][0:1:888],wind_speed[0:1:0][0:1:0][0:1:1488][0:1:888],precipitation_amount_acc[0:1:0][0:1:0][0:1:1488][0:1:888],snowfall_amount_acc[0:1:0][0:1:0][0:1:1488][0:1:888]"
        path = org_path + date.strftime("%Y/%m/%d") + "/" + h_lookup[hour_str][0] + "/fc" + date.strftime("%Y%m%d") + h_lookup[hour_str][0] + "_0" + h_lookup[hour_str][1] + "_fp.nc"+ addon
        
        # Downloads are prone to fail, thus some light error handling here.
        try:
            error_count = 0
            DS = xr.open_dataset(path)

        except: # we will try to download 5 times, then just insert the data as missing values and move on.
            error_count+=1
            if(error_count < 5):
                print("Download of file" + path + " did not work. Attempting again.")
                
                time.sleep(20)
                DS = xr.open_dataset(path)
            else:
                print("Download of file" + path + " did not work. Giving up.")
                print(self.state)
                raise LookupError
                # print("Download of file" + path + " did not work. Inserting as Missing.")
                # A = np.zeros((len(self.vars), len(self.latlons))) * np.nan
                # return A
         
        x = DS.variables["x"].data
        y = DS.variables["y"].data
        proj = pyproj.Proj(DS.variables["projection_lambert"].attrs["proj4"])
        A = np.zeros((len(self.vars), len(self.latlons))) * np.nan

        for varidx, var in enumerate(self.vars):
            a = DS.variables[var].data[0, 0, :, :]
            assert a.shape == (1489, 889), "as expected. {0}".format(DS.variables[var].data.shape)
            for ptidx, latlon in enumerate(self.latlons):
                lat = latlon[0]
                lon = latlon[1]
                X, Y = proj(lon, lat)
                Ix = np.argmin(np.abs(x - X))
                Iy = np.argmin(np.abs(y - Y))
                
                A[varidx, ptidx] = a[Iy, Ix]

        assert sum(sum(np.isnan(A)))==0
        return A.tolist()


    def download_nora3_parallel(self):

        timerange = pd.date_range(start=self.start_date, end=self.end_date, freq='H')
        
        if self.sampling:
            timerange = timerange[timerange.day == 15]
        
        
        upload_time = ['00', '06', '12', '18']
        casting_set = ['04', '05', '06', '07', '08', '09']
        shifted_range = range(4, 24)
        wrapped_range = range(0, 4)
        upload_permutations = [[hour6, hour] for hour6 in upload_time for hour in casting_set]
        h_lookup = {str(hour).zfill(2): element for hour, element in zip(shifted_range, upload_permutations)}
        h_lookup.update({str(hour).zfill(2): element for hour, element in zip(wrapped_range, upload_permutations[-4:])})

        #self.save_range = self.save_range + 1
        save_counter = 0
        export = np.zeros((len(self.vars), len(self.latlons), self.save_range)) * np.nan

        #pool = multiprocessing.Pool(processes = 1) # processes = 4
        
        #results = []

        for i,hour in enumerate(timerange):
            #results.append(pool.apply_async(self.process_hour, (hour, h_lookup)))
            #results.append(self.process_hour(hour, h_lookup))
            A = np.array(self.process_hour(hour, h_lookup))
        #for i, result in enumerate(results):
            #A = result.get()
            if save_counter % (self.save_range) == 0 and save_counter!=0:
                filename = "nora3_{0}_{2}hours_{1}vars_{3}latlons.npy".format(timerange[i].strftime("%Y%m%d_%H"), len(self.vars), self.save_range, len(self.latlons))
                assert sum(sum(sum(np.isnan(export))))==0
                np.save(filename, export)
                print(filename)
                export = np.zeros((len(self.vars), len(self.latlons), self.save_range)) * np.nan
                export[:, :, save_counter % self.save_range] = A
            else:
                export[:, :, save_counter % self.save_range] = A
            save_counter +=1




def Sunpoint_import(path = os.path.dirname(os.getcwd()) + "/code/data")-> tuple:
    """
    returns start_date, end_date, timerange and a pd.dataframe of Sunpoint
    only downloads sunpoint if it isnt in the working directory already with the name 'sunpoint_data.csv'
    """
    loc = "https://thredds.met.no/thredds/dodsC/sunpoint/CollectedMeasurements/rsds_flagged_1hr_selection_v5_2016-2020.nc"
    if os.path.exists(path):
        with open(path+"/sunpoint_data.csv","rb") as file:
            df = pd.read_csv(file,index_col=[0,1])

    else:   
        dataset = xr.open_dataset(loc)
        df = dataset.to_dataframe()

        with open(path, 'wb') as file:
            df.to_csv(file)
    start_date = "2016-01-01 00:30:00"
    end_date = "2020-12-31 23:30:00"    

    timerange = pd.date_range(start=start_date, end = end_date, freq = '1h')
    return start_date,end_date,timerange,df


def NORA_import(start:str, end:str, path: str, dims: tuple) -> tuple:
    """
    This function opens the downloaded nora files and concatenates them for a given timerange
    It recognizes when they are missing files. In that case it will give a printout, 
    save the missing label in a second output array, and insert nan values in the concatenated file.

    example input:

    #path = ""

    #end = '2021-04-01'

    #start = '2021-02-01'

    #dims = (8,20,6)
    """

    files = [file for file in os.listdir(path) if file.endswith(".npy") and file.startswith("nora3_2")]
    try:
        timerange = pd.date_range(start=start, end=end, freq='6H')[1:] # skip the first instance, as this would cover data before and up to the start.
    except:
        raise "Didn't provide a valid start or end date."
    missing = []
    data = None
    filled = False
    for instance in timerange:
        dt_updated = instance #- timedelta(hours=1)
        key = dt_updated.strftime("%Y%m%d_%H")
        search = np.char.find(files,key ) != -1
        #print(sum(search))
        if sum(search) !=0:
            file = files[np.where(search)[0][0]]
            #print(path+ "/" + file)
            
            dtemp = np.load(path+ "/" + file)
            #print(dtemp)
            if filled == False:
                data= dtemp
                filled = True
            else: 
                data = np.concatenate([data,dtemp],axis =2)
        else:
            print("The file for instance " +str(instance) + " with key " +key+ "is missing!")
            #print(search)
            missing.append([instance, key])
            miss = np.zeros(dims)*np.nan
            if filled == False:
                data= miss
                filled = True
            else: 
                data = np.concatenate([data,miss],axis =2)
        #print(address)
        #dat = np.load(path+'/'+address)

    return data, missing




# class CAN_wrapper(cdsapi.Client):
#     """
#     Wraps around the cdsapi. Only works with it though, so needs the regular installations steps provided by cdsapi.
#     """
#     def __init__(self,name: str, latlons: list, start, stop: str, time:str, step= '15minute', alt="-999."):
#         """
#         Need date start and stop format as yyyy-mm-dd
        
#         """
#         super().__init__()
#         self.data_name = name
#         self.latlons = latlons
#         self.altitude = alt
#         self.start = start
#         self.stop = stop
#         self.date = start + "/" + stop
#         self.step = step
#         self.time = time

#     def gen_files(self):
#         for lat,lon in self.latlons:
#             self.download(lat,lon)
    
#     def download(self,lat,lon):
    
#         filename = 'can_' + self.start+"_"+self.stop +"_"+ str(int(lat*10000))+"_"+str(int(lon*10000)) + '.nc'
#         self.retrieve(
#         self.data_name,
#         {
#             'location': {
#                 'latitude': lat,
#                 'longitude': lon,
#             },
#             'altitude': self.altitude,
#             'date': self.date,
#             'time_step': self.step,
#             'time_reference': self.time,
#             'format': 'netcdf',
#             'sky_type': 'observed_cloud',
#         },
#         filename)
        
#     def concat_files(self,path):
#         files = [file for file in os.listdir(path) if file.endswith(".nc") and file.startswith("can_")]
        
#         for i,file in enumerate(files):
#             DS = xr.open_dataset(path+"/"+file)
#             #print(file)
#             if i==0:
#                 data = np.zeros((2,len(files), len(DS.time.data))) * np.nan
#                 time = np.empty((len(files), len(DS.time.data)), dtype="datetime64[s]") 
#                 time[:,:] = np.datetime64('NaT')
#                 #print(data.shape)
#             t = DS.time.data
#             ghi=DS.GHI.data[:,0,0,0]
#             cs=DS.CLEAR_SKY_GHI.data[:,0,0,0]
#             #print(t.shape)
#             time[i,:] = t
#             data[0,i,:] = ghi
#             data[1,i,:] = cs
#             print(time)
    
#         filename = "can_con_"+self.start+"_"+self.stop+".npz"
#         np.savez(filename,time = time, ghi = data[0], cs = data[1])


def CAN_import(path = os.path.join(os.path.dirname(os.getcwd()), "/code/data"))-> np.array:
    try:
        file = [file for file in os.listdir(path) if file.startswith("can_con_")][0]
        data = np.load(path+"/"+file)
        print("Returning file: " + file)
        return data
    except:
        print(path)
        assert os.path.isdir(path), NotADirectoryError
        raise FileNotFoundError
    

def sliding_window_sum(arr, window_size, start):
    """ Meant to accumulate values before a point in time."""
    zeros= np.zeros((20,window_size-(1+start)))
    padded_arr = np.concatenate((zeros, arr), axis=1)
    #print(padded_arr.shape)
    window_view = np.lib.stride_tricks.sliding_window_view(padded_arr, window_shape=window_size, axis = 1)
    #print(window_view.shape)
    #print(window_view[:,::window_size,:])
    window_sum = np.sum(window_view[:,::window_size,:], axis=2)

    return window_sum