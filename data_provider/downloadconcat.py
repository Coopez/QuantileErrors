

import numpy as np
import pandas as pd
import os

from  datetime import timedelta

start ='2021-02-01'
end = '2021-02-28'
path = r"C:\Users\nikla\Documents\phd\phase_1\code\data"

dims = (8,20,6)


files = [file for file in os.listdir(path) if file.endswith(".npy")]
try:
    timerange = pd.date_range(start=start, end=end, freq='6H')[1:] # skip the first instance, as this would cover data before and up to the start.
except:
    raise "Didn't provide a valid start or end date."
missing = []
data = None
filled = False
for instance in timerange:
    dt_updated = instance - timedelta(hours=1)
    key = dt_updated.strftime("%Y%m%d_%H")
    search = np.char.find(files,key ) != -1
    if sum(search) !=0:
        file = files[np.where(search)[0][0]]

        dtemp = np.load(path+ "\\" + file)
        if filled == False:
            data= dtemp
            filled = True
        else: 
            data = np.concatenate([data,dtemp],axis =2)
    else:
        print("The file for instance " +str(instance) + " with key " +key+ "is missing!")
        missing.append([instance, key])
        miss = np.zeros(dims)*np.nan
        if filled == False:
            data= miss
            filled = True
        else: 
            data = np.concatenate([data,miss],axis =2)
    #print(address)
    #dat = np.load(path+'/'+address)
