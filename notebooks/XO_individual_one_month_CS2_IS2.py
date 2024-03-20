# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 17:19:31 2024

@author: rmfha
"""




def CRYO2ICE_identify_XO(df_CS2, df_IS2, dist_req):
    '''
    
    '''
    
    from sklearn.neighbors import BallTree
    import numpy as np
    import pandas as pd

    query_lats = df_CS2[['lat']].to_numpy()
    query_lons = df_CS2[['lon']].to_numpy()

    tree = BallTree(np.deg2rad(df_IS2[['lat', 'lon']].values),  metric='haversine')

    dist_in_metres = dist_req
    earth_radius_in_metres = 6371*1000
    radius = dist_in_metres/earth_radius_in_metres

    is_within, distances = tree.query_radius(np.deg2rad(np.c_[query_lats, query_lons]), r=radius, count_only=False, return_distance=True) 
    distances_in_metres = distances*earth_radius_in_metres
    
    
    RMSD_IS2_fb_A, RMSD_IS2_fb_B = np.empty(len(query_lats)),np.empty(len(query_lats))
    RMSD_CS2_fb_A, RMSD_CS2_fb_B = np.empty(len(query_lats)),np.empty(len(query_lats))
    RMSD_CS2_LARM_fb_A, RMSD_CS2_LARM_fb_B = np.empty(len(query_lats)),np.empty(len(query_lats))
    RMSD_CS2_CCI_fb_A, RMSD_CS2_CCI_fb_B = np.empty(len(query_lats)),np.empty(len(query_lats))
    RMSD_C2I_fb_A, RMSD_C2I_fb_B = np.empty(len(query_lats)),np.empty(len(query_lats))
    RMSD_C2I_LARM_fb_A, RMSD_C2I_LARM_fb_B = np.empty(len(query_lats)),np.empty(len(query_lats))
    RMSD_C2I_CCI_fb_A, RMSD_C2I_CCI_fb_B = np.empty(len(query_lats)),np.empty(len(query_lats))
    lat_A, lon_A, lat_B, lon_B = np.empty(len(query_lats)),np.empty(len(query_lats)),np.empty(len(query_lats)),np.empty(len(query_lats))
    ID_A, ID_B = [],[]
    
    k = 0
    
    for i in is_within:
        if len(i)>5:
            data_fb_IS2 = df_IS2['IS2_w_mean_fb_MSS'].iloc[i]
            data_fb_CS2 = df_IS2['CS2_smooth'].iloc[i]
            data_fb_CS2_LARM = df_IS2['LARM_rfb_smooth'].iloc[i]
            data_fb_CS2_CCI = df_IS2['CCI_rfb_smooth'].iloc[i]
            data_fb_C2I = df_IS2['snow_depth_original_MSS_smoothed'].iloc[i]
            data_fb_C2I_LARM = df_IS2['snow_depth_original_MSS_LARM_smoothed'].iloc[i]
            data_fb_C2I_CCI = df_IS2['snow_depth_original_MSS_CCI_smoothed'].iloc[i]
            lat = df_IS2['lat'].iloc[i]
            lon = df_IS2['lon'].iloc[i]
            ID = df_IS2['ID'].iloc[i[0]]

            RMSD_IS2_fb_A_mean = np.nanmean(data_fb_IS2)
            RMSD_CS2_fb_A_mean  = np.nanmean(data_fb_CS2)
            RMSD_CS2_LARM_fb_A_mean  = np.nanmean(data_fb_CS2_LARM)
            RMSD_CS2_CCI_fb_A_mean  = np.nanmean(data_fb_CS2_CCI)
            RMSD_C2I_fb_A_mean  = np.nanmean(data_fb_C2I)
            RMSD_C2I_LARM_fb_A_mean  = np.nanmean(data_fb_C2I_LARM)
            RMSD_C2I_CCI_fb_A_mean  = np.nanmean(data_fb_C2I_CCI)
            lat_x = np.nanmean(lat)
            lon_x = np.nanmean(lon)
            
        else:
            RMSD_IS2_fb_A_mean  = np.nan
            RMSD_CS2_fb_A_mean  = np.nan
            RMSD_CS2_LARM_fb_A_mean  = np.nan
            RMSD_CS2_CCI_fb_A_mean  = np.nan
            RMSD_C2I_fb_A_mean  = np.nan
            RMSD_C2I_LARM_fb_A_mean  = np.nan
            RMSD_C2I_CCI_fb_A_mean  = np.nan
            lat_x = np.nan
            lon_x = np.nan
            ID = np.nan

        
        RMSD_IS2_fb_A[k] = RMSD_IS2_fb_A_mean
        RMSD_CS2_fb_A[k]  = RMSD_CS2_fb_A_mean
        RMSD_CS2_LARM_fb_A[k]  = RMSD_CS2_LARM_fb_A_mean
        RMSD_CS2_CCI_fb_A[k]  = RMSD_CS2_CCI_fb_A_mean
        RMSD_C2I_fb_A[k] = RMSD_C2I_fb_A_mean
        RMSD_C2I_LARM_fb_A[k]  = RMSD_C2I_LARM_fb_A_mean
        RMSD_C2I_CCI_fb_A[k]  = RMSD_C2I_CCI_fb_A_mean
        lat_A[k] = lat_x
        lon_A[k] = lon_x
        ID_A = np.append(ID_A, ID)
        
        
        k = k+1
    
    #print(k)
    query_lats = df_CS2[['lat']].to_numpy()       
    query_lons = df_CS2[['lon']].to_numpy()
    tree = BallTree(np.deg2rad(df_CS2[['lat', 'lon']].values),  metric='haversine')

    dist_in_metres = dist_req
    earth_radius_in_metres = 6371*1000
    radius = dist_in_metres/earth_radius_in_metres

    is_within, distances = tree.query_radius(np.deg2rad(np.c_[query_lats, query_lons]), r=radius, count_only=False, return_distance=True) 
    distances_in_metres = distances*earth_radius_in_metres


    k = 0
    
    for i in is_within:
        if len(i)>5:
            data_fb_IS2 = df_CS2['IS2_w_mean_fb_MSS'].iloc[i]
            data_fb_CS2 = df_CS2['CS2_smooth'].iloc[i]
            data_fb_CS2_LARM = df_CS2['LARM_rfb_smooth'].iloc[i]
            data_fb_CS2_CCI = df_CS2['CCI_rfb_smooth'].iloc[i]
            data_fb_C2I = df_CS2['snow_depth_original_MSS_smoothed'].iloc[i]
            data_fb_C2I_LARM = df_CS2['snow_depth_original_MSS_LARM_smoothed'].iloc[i]
            data_fb_C2I_CCI = df_CS2['snow_depth_original_MSS_CCI_smoothed'].iloc[i]
            lat = df_CS2['lat'].iloc[i]
            lon = df_CS2['lon'].iloc[i]
            ID = df_CS2['ID'].iloc[i[0]]
            

            RMSD_IS2_fb_B_mean = np.nanmean(data_fb_IS2)
            RMSD_CS2_fb_B_mean  = np.nanmean(data_fb_CS2)
            RMSD_CS2_LARM_fb_B_mean  = np.nanmean(data_fb_CS2_LARM)
            RMSD_CS2_CCI_fb_B_mean  = np.nanmean(data_fb_CS2_CCI)
            RMSD_C2I_fb_B_mean  = np.nanmean(data_fb_C2I)
            RMSD_C2I_LARM_fb_B_mean  = np.nanmean(data_fb_C2I_LARM)
            RMSD_C2I_CCI_fb_B_mean  = np.nanmean(data_fb_C2I_CCI)
            lat_x = np.nanmean(lat)
            lon_x = np.nanmean(lon)
            
            
        else:
            RMSD_IS2_fb_B_mean  = np.nan
            RMSD_CS2_fb_B_mean  = np.nan
            RMSD_CS2_LARM_fb_B_mean  = np.nan
            RMSD_CS2_CCI_fb_B_mean  = np.nan
            RMSD_C2I_fb_B_mean  = np.nan
            RMSD_C2I_LARM_fb_B_mean  = np.nan
            RMSD_C2I_CCI_fb_B_mean  = np.nan
            ID = np.nan
        


        
        RMSD_IS2_fb_B[k] = RMSD_IS2_fb_B_mean
        RMSD_CS2_fb_B[k]  = RMSD_CS2_fb_B_mean
        RMSD_CS2_LARM_fb_B[k]  = RMSD_CS2_LARM_fb_B_mean
        RMSD_CS2_CCI_fb_B[k]  = RMSD_CS2_CCI_fb_B_mean
        RMSD_C2I_fb_B[k] = RMSD_C2I_fb_B_mean
        RMSD_C2I_LARM_fb_B[k]  = RMSD_C2I_LARM_fb_B_mean
        RMSD_C2I_CCI_fb_B[k]  = RMSD_C2I_CCI_fb_B_mean
        lat_B[k] = lat_x
        lon_B[k] = lon_x
        ID_B = np.append(ID_B, ID)
        
        k = k+1
    
    #print(k)
    df_IS2_new = pd.DataFrame({'IS2_fb_A':RMSD_IS2_fb_A,
                               'IS2_fb_B':RMSD_IS2_fb_B,
                               'CS2_fb_A':RMSD_CS2_fb_A,
                               'CS2_fb_B':RMSD_CS2_fb_B,
                               'CS2_LARM_fb_A':RMSD_CS2_LARM_fb_A,
                               'CS2_LARM_fb_B':RMSD_CS2_LARM_fb_B,
                               'CS2_CCI_fb_A':RMSD_CS2_CCI_fb_A,
                               'CS2_CCI_fb_B':RMSD_CS2_CCI_fb_B,
                               'C2I_A':RMSD_C2I_fb_A,
                               'C2I_B':RMSD_C2I_fb_B,
                               'C2I_LARM_A':RMSD_C2I_LARM_fb_A,
                               'C2I_LARM_B':RMSD_C2I_LARM_fb_B,
                               'C2I_CCI_A':RMSD_C2I_CCI_fb_A,
                               'C2I_CCI_B':RMSD_C2I_CCI_fb_B,
                               'lat_A':lat_A,
                               'lon_A':lon_A,
                               'lat_B':lat_B,
                               'lon_B':lon_B, 
                               'ID_A':ID_A,
                               'ID_B':ID_B
                               
        
        
        
    })

    
    return df_IS2_new


import datetime
def load_CRYO2ICE_XO(files_check_CS2_data):
    li = []
    k = 0
    for filename in files_check_CS2_data:
        df = pd.read_csv(filename, index_col=None, header=0)
        basename_without_ext = os.path.splitext(os.path.basename(filename))[0]
        txt1 = basename_without_ext.split('_')[8]
        time_n1 = datetime.datetime.strptime(txt1,"%Y%m%dT%H%M%S")
        df['ID']=time_n1

        li.append(df)
        k = k+1
    frame2 = pd.concat(li, axis=0, ignore_index=True)
   
    return frame2
    
    
#%% initialisation
import os 
import numpy as np 
import proplot as pplt
import h5py 
import pandas as pd
import sys
import datetime
from matplotlib.colors import LogNorm
from scipy import signal
import datetime
from netCDF4 import Dataset
    
    
#%% Load IS2 data from server 



yr_folder='2020'
month_folder = '03'

directory_path = r'Z:\rawdata\ICESAT-2\ATL10\rel005\ARCTIC'+'/'+yr_folder+'/' +month_folder
file_extension = '.h5'
files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.endswith(file_extension)]


#%% CS2 Baseline-E

def load_CRYO2ICE_XO(files_check_CS2_data):
    li = []
    k = 0
    for filename in files_check_CS2_data:
        basename_without_ext = os.path.splitext(os.path.basename(filename))[0]
        txt1 = basename_without_ext.split('_')[8]
        time_n1 = datetime.datetime.strptime(txt1,"%Y%m%dT%H%M%S")
        file_in = Dataset(filename,"r",format="NETCDF4")
        lat, lon = file_in.variables['lat_poca_20_ku'][:], file_in.variables['lon_poca_20_ku'][:]
        frb = file_in.variables['radar_freeboard_20_ku'][:]
        df = pd.DataFrame({'lat':lat.flatten(), 'lon':lon.flatten(), 'frb':frb.flatten()})     
        
        tname = "time_20_ku"
        nctime = file_in.variables[tname][:] # get values
        t_unit = file_in.variables[tname].units # get unit  "days since 1950-01-01T00:00:00Z"
        t_cal = file_in.variables[tname].calendar
        tvalue = num2date(nctime,units = t_unit,calendar = t_cal)
        #str_time = [str(i) for i in tvalue]
        df['ID']=tvalue
        
        df = df[df['lat']>60]

        li.append(df)
        k = k+1
    frame2 = pd.concat(li, axis=0, ignore_index=True)
   
    return frame2

def CRYO2ICE_identify_XO(df_CS2, df_IS2, dist_req, t_unit, t_cal):
    
    from sklearn.neighbors import BallTree
    import numpy as np
    import pandas as pd

    query_lats = df_CS2[['lat']].to_numpy()
    query_lons = df_CS2[['lon']].to_numpy()

    tree = BallTree(np.deg2rad(df_IS2[['lat', 'lon']].values),  metric='haversine')

    dist_in_metres = dist_req
    earth_radius_in_metres = 6371*1000
    radius = dist_in_metres/earth_radius_in_metres

    is_within, distances = tree.query_radius(np.deg2rad(np.c_[query_lats, query_lons]), r=radius, count_only=False, return_distance=True) 
    distances_in_metres = distances*earth_radius_in_metres
    
    
    RMSD_CS2_fb_A, RMSD_CS2_fb_B = np.empty(len(query_lats)),np.empty(len(query_lats))
    lat_A, lon_A, lat_B, lon_B = np.empty(len(query_lats)),np.empty(len(query_lats)),np.empty(len(query_lats)),np.empty(len(query_lats))
    ID_A, ID_B = [],[]
    
    k = 0
    
    for i in is_within:
        if len(i)>=1:
            data_fb_CS2 = df_IS2['frb'].iloc[i]
            lat = df_IS2['lat'].iloc[i]
            lon = df_IS2['lon'].iloc[i]
            ID = df_IS2['ID'].iloc[i]

            
            RMSD_CS2_fb_A_mean  = np.nanmean(data_fb_CS2)
            lat_x = np.nanmean(lat)
            lon_x = np.nanmean(lon)
            
            datetime_array = np.array(ID)
            import cftime

            # Convert the array to numeric representation
            numeric_array = cftime.date2num(datetime_array, t_unit, t_cal)

            # Compute the average numeric representation
            average_numeric = np.mean(numeric_array)

            # Convert the average numeric representation back to datetime
            average_datetime = cftime.num2date(average_numeric, t_unit, t_cal)
            ID  = average_datetime
            
        else:
            RMSD_CS2_fb_A_mean  = np.nan
            lat_x = np.nan
            lon_x = np.nan
            ID = np.nan

        
        RMSD_CS2_fb_A[k]  = RMSD_CS2_fb_A_mean
        lat_A[k] = lat_x
        lon_A[k] = lon_x
        ID_A = np.append(ID_A, ID)
        
        
        k = k+1
    
    #print(k)
    query_lats = df_CS2[['lat']].to_numpy()       
    query_lons = df_CS2[['lon']].to_numpy()
    tree = BallTree(np.deg2rad(df_CS2[['lat', 'lon']].values),  metric='haversine')

    dist_in_metres = dist_req
    earth_radius_in_metres = 6371*1000
    radius = dist_in_metres/earth_radius_in_metres

    is_within, distances = tree.query_radius(np.deg2rad(np.c_[query_lats, query_lons]), r=radius, count_only=False, return_distance=True) 
    distances_in_metres = distances*earth_radius_in_metres


    k = 0
    
    for i in is_within:
        if len(i)>=1:
            data_fb_CS2 = df_CS2['frb'].iloc[i]
            lat = df_CS2['lat'].iloc[i]
            lon = df_CS2['lon'].iloc[i]
            ID = df_CS2['ID'].iloc[i]

            
            RMSD_CS2_fb_B_mean  = np.nanmean(data_fb_CS2)
            lat_x = np.nanmean(lat)
            lon_x = np.nanmean(lon)
            
            datetime_array = np.array(ID)
            import cftime

            # Convert the array to numeric representation
            numeric_array = cftime.date2num(datetime_array, t_unit, t_cal)

            # Compute the average numeric representation
            average_numeric = np.mean(numeric_array)

            # Convert the average numeric representation back to datetime
            average_datetime = cftime.num2date(average_numeric, t_unit, t_cal)
            ID  = average_datetime
            
            
        else:
            RMSD_CS2_fb_B_mean  = np.nan
            ID = np.nan
        


        RMSD_CS2_fb_B[k]  = RMSD_CS2_fb_B_mean
        lat_B[k] = lat_x
        lon_B[k] = lon_x
        ID_B = np.append(ID_B, ID)
        
        k = k+1
    
    #print(k)
    df_IS2_new = pd.DataFrame({'CS2_fb_A':RMSD_CS2_fb_A,
                               'CS2_fb_B':RMSD_CS2_fb_B,
                               'lat_A':lat_A,
                               'lon_A':lon_A,
                               'lat_B':lat_B,
                               'lon_B':lon_B, 
                               'ID_A':ID_A,
                               'ID_B':ID_B
                               
        
        
        
    })

    
    return df_IS2_new



import warnings
warnings.filterwarnings("ignore")

analysis = 'Baseline-E'
satellite = 'CS2'
directory_path = r'D:\DTU\CRYO2ICE\XO\CS2_BaselineE_XO'
file_extension = '.nc'
files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.endswith(file_extension)]
with open("{}\BaselineE_XO_print_v2.txt".format(directory_path),"w") as f:
    time_difference = [1, 3, 6, 12, 24, 48]
    distance = [150, 750, 1500, 3500]
    
    #time_difference = [3]
    #distance = [750]
    #for i in np.arange(0,len(files)):
    for delta_t in time_difference: 
        di = '###### ... Analysis of Baseline-E data with delta_t = {} hrs for March 2021.\n'.format(delta_t)
        print(di)
        f.write(di)
        for dist_req in distance: 
            li = []
            for i in np.arange(0,len(files)-1):
                cross_over_files = []
                fp1 = files[i]
                basename_without_ext = os.path.splitext(os.path.basename(fp1))[0]
                txt1a = basename_without_ext.split('_')[7]
                txt1b = basename_without_ext.split('_')[8]
                time_n1a = datetime.datetime.strptime(txt1a,"%Y%m%dT%H%M%S")
                time_n1b = datetime.datetime.strptime(txt1b,"%Y%m%dT%H%M%S")
            #try:
                for j in np.arange(i+1,len(files)-1):
                    fp2 = files[j]
                    basename_without_ext = os.path.splitext(os.path.basename(fp2))[0]
                    txt2a = basename_without_ext.split('_')[7]
                    txt2b = basename_without_ext.split('_')[8]
                    time_n2a = datetime.datetime.strptime(txt2a,"%Y%m%dT%H%M%S")
                    time_n2b = datetime.datetime.strptime(txt2b,"%Y%m%dT%H%M%S")
                    diff_ta = abs((time_n1a - time_n2a)/datetime.timedelta(hours=1))
                    diff_tb = abs((time_n1b - time_n2b)/datetime.timedelta(hours=1))
                    diff_tc = abs((time_n1a - time_n2b)/datetime.timedelta(hours=1))
                    diff_td = abs((time_n1b - time_n2a)/datetime.timedelta(hours=1))
                    if (diff_ta < delta_t) or (diff_tb < delta_t) or (diff_tc < delta_t) or (diff_td < delta_t):
                        cross_over_files = np.append(cross_over_files, fp2)
                #        print(diff_t)
                #print(cross_over_files)
            
                if len(cross_over_files)>1:
                    file_in = Dataset(fp1,"r",format="NETCDF4")
                    lat, lon = file_in.variables['lat_poca_20_ku'][:], file_in.variables['lon_poca_20_ku'][:]
                    frb = file_in.variables['radar_freeboard_20_ku'][:]
                    df_original = pd.DataFrame({'lat':lat.flatten(), 'lon':lon.flatten(), 'frb':frb.flatten()})

                    from netCDF4 import Dataset,num2date
                    
                    tname = "time_20_ku"
                    nctime = file_in.variables[tname][:] # get values
                    t_unit = file_in.variables[tname].units # get unit  "days since 1950-01-01T00:00:00Z"
                    t_cal = file_in.variables[tname].calendar
                    tvalue = num2date(nctime,units = t_unit,calendar = t_cal)
                    #str_time = [str(i) for i in tvalue]
                    
                    df_original['ID']=tvalue
                    df_original = df_original[df_original['lat']>60]
                    df_XO = load_CRYO2ICE_XO(cross_over_files)
                    df_XO_new = CRYO2ICE_identify_XO(df_original, df_XO, dist_req, t_unit, t_cal)
                    li.append(df_XO_new)
            
            df_XO_total = pd.concat(li, axis=0, ignore_index=True)
            try:
                def rmse(predictions, targets):
                    return np.sqrt(((predictions - targets) ** 2).mean())
                
                df_XO_total_nonna = df_XO_total[(df_XO_total['CS2_fb_A'].notna())]
                df_XO_total_nonna = df_XO_total_nonna[(df_XO_total_nonna['CS2_fb_B'].notna())]
                df_XO_total_nonna['time_delta']= abs((df_XO_total_nonna['ID_A'] - df_XO_total_nonna['ID_B'])/datetime.timedelta(hours=1))
                df_XO_total_nonna['abs_diff_A_B']= abs((df_XO_total_nonna['CS2_fb_A'] - df_XO_total_nonna['CS2_fb_B']))
                
                if len(df_XO_total_nonna)>1:
                    rmse = rmse(df_XO_total_nonna['CS2_fb_A'],df_XO_total_nonna['CS2_fb_B'])*100
                    
                    df_XO_total_nonna.to_csv(directory_path+'/XO_differences_{}_{}_{}hrs_{}m.csv'.format(satellite, analysis, delta_t, dist_req))
                    
                    di = 'SUCCES ANALYSIS! For delta_t = {} hrs. and distance req. = {} m, N of XOs analysed = {} and RMSE = {:.2f} cm\n'.format(delta_t, dist_req, len(df_XO_total_nonna), rmse)
                    print(di)
                    f.write(di)
                    
                    import proplot as pplt
                    import matplotlib.pyplot as plt
                    import cartopy
                    import cartopy.feature as cfeature
                    import cartopy.crs as ccrs
                    resol='110m'
                    fig,ax = pplt.subplots(refwidth=2.5, ncols=2, nrows=1, sharex=False, proj={2:'nplaea'})
                    fig.patch.set_facecolor('white')
                    
                    axs = ax[0]
                    axs.scatter(df_XO_total_nonna['time_delta'], df_XO_total_nonna['abs_diff_A_B'], c='grey')
                    txt_fig = 'max abs. diff(XOs) = {:.2f} cm \navg. abs. diff(XOs) = {:.2f} cm\nRMSE = {:.2f} cm\nN = {}\navg. $\Delta$t = {:.2f} hrs.'.format(np.nanmax(df_XO_total_nonna['abs_diff_A_B'])*100,np.nanmean(df_XO_total_nonna['abs_diff_A_B'])*100, rmse, len(df_XO_total_nonna), np.nanmean(df_XO_total_nonna['time_delta']))
                    axs.format(ultitle=txt_fig,title='{}: $\Delta$t = {} hrs., dist. req. = {} m'.format(analysis, delta_t, dist_req))
                    axs.format(xlim=[0, delta_t],xlabel='$\Delta$t (hrs)', ylabel = 'abs. diff. between  {} freeboards at XOs (m)'.format(satellite))
                    
                    
                
                    axs = ax[1]
                    im = axs.scatter(df_XO_total_nonna['lon_A'], df_XO_total_nonna['lat_B'], c=df_XO_total_nonna['time_delta'], cmap='inferno_r', s=0.5, vmin=0, vmax=delta_t)
                    cb = axs.colorbar(im, label='$\Delta$t (hrs.)', loc='b')
                    axs.add_feature(cfeature.LAND, facecolor='lightgrey')
                    axs.coastlines(resolution=resol, color='k')
                    axs.format(boundinglat=60)
    
                    plt.savefig(directory_path+'/XO_differences_{}_{}_{}hrs_{}m.png'.format(satellite, analysis, delta_t, dist_req),dpi=300)
                else: 
                    di = 'ERROR! ... RMSE cannot be calculated for: delta_t = {} hrs. and dist. req = {} m, since there where only N = {} observations.\n'.format(delta_t, dist_req, len(df_XO_total_nonna))
                    print(di)
                    f.write(di)
            except: 
                di = 'ERROR! ... RMSE cannot be calculated for: delta_t = {} hrs. and dist. req = {} m, since there where only N = {} observations.\n'.format(delta_t, dist_req, len(df_XO_total_nonna))
                print(di)
                f.write(di)
                continue
        print('\n')
        f.write('\n')


#%%

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

directory_path = r'D:\DTU\CRYO2ICE\XO\CS2_BaselineE_XO'
file_extension = '.csv'
files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.endswith(file_extension)]

delta_t_val = [1, 3, 6, 12, 24, 48]
MAD_150m, MAD_750m, MAD_1500m, MAD_3500m = np.zeros(len(delta_t_val))*np.nan,np.zeros(len(delta_t_val))*np.nan,np.zeros(len(delta_t_val))*np.nan,np.zeros(len(delta_t_val))*np.nan
N_150m, N_750m, N_1500m, N_3500m = np.zeros(len(delta_t_val))*np.nan,np.zeros(len(delta_t_val))*np.nan,np.zeros(len(delta_t_val))*np.nan,np.zeros(len(delta_t_val))*np.nan
RMSE_150m, RMSE_750m, RMSE_1500m, RMSE_3500m = np.zeros(len(delta_t_val))*np.nan,np.zeros(len(delta_t_val))*np.nan,np.zeros(len(delta_t_val))*np.nan,np.zeros(len(delta_t_val))*np.nan


analysis = 'Baseline-E'
satellite = 'CS2'
i_dir = r'D:\DTU\CRYO2ICE\XO\CS2_BaselineE_XO'
#i_fn = 'XO_differences_CS2_Baseline-E_48hrs_3500m.csv'
with open("{}\BaselineE_XO_print_MAD_statistics.txt".format(directory_path),"w") as f:
    for i_fn in files:
        basename_without_ext = os.path.splitext(os.path.basename(i_fn))[0]
        try:
            txt1a = basename_without_ext.split('_')[4][0:2]
            delta_t = int(txt1a)
        except:
            txt1a = basename_without_ext.split('_')[4][0]
            delta_t = int(txt1a)
        try:
            txt1b = basename_without_ext.split('_')[5][0:4]
            dist_req = int(txt1b)
        except:
            txt1b = basename_without_ext.split('_')[5][0:3]
            dist_req = int(txt1b)
        df_XO_total_nonna = pd.read_csv(i_fn)
        
        rmse_val = rmse(df_XO_total_nonna['CS2_fb_A'],df_XO_total_nonna['CS2_fb_B'])*100
        
        from statsmodels import robust
        data = df_XO_total_nonna['abs_diff_A_B']
        MAD = robust.mad(data, c=1)*100
        di = 'Median absolute deviation of {} {} XO analysis with {} hrs. and {} m as dist. req = {:.2f} cm\n'.format(satellite, analysis, delta_t, dist_req, MAD)
        print(di)
        f.write(di)
        
        '''
        import proplot as pplt
        import matplotlib.pyplot as plt
        import cartopy
        import cartopy.feature as cfeature
        import cartopy.crs as ccrs
        resol='110m'
        fig,ax = pplt.subplots(refwidth=2.5, ncols=2, nrows=1, sharex=False, proj={2:'nplaea'})
        fig.patch.set_facecolor('white')
        
        axs = ax[0]
        axs.scatter(df_XO_total_nonna['time_delta'], df_XO_total_nonna['abs_diff_A_B'], c='grey')
        txt_fig = 'max abs. diff(XOs) = {:.2f} cm \navg. abs. diff(XOs) = {:.2f} cm\nMAD = {:.2f} cm\nRMSE = {:.2f} cm\nN = {}\navg. $\Delta$t = {:.2f} hrs.'.format(np.nanmax(df_XO_total_nonna['abs_diff_A_B'])*100,np.nanmean(df_XO_total_nonna['abs_diff_A_B'])*100, MAD,rmse_val, len(df_XO_total_nonna), np.nanmean(df_XO_total_nonna['time_delta']))
        axs.format(ultitle=txt_fig,title='{}: $\Delta$t = {} hrs., dist. req. = {} m'.format(analysis, delta_t, dist_req))
        axs.format(xlim=[0, delta_t],xlabel='$\Delta$t (hrs)', ylabel = 'abs. diff. between  {} freeboards at XOs (m)'.format(satellite))
        
        
        
        axs = ax[1]
        im = axs.scatter(df_XO_total_nonna['lon_A'], df_XO_total_nonna['lat_B'], c=df_XO_total_nonna['time_delta'], cmap='inferno', s=0.5, vmin=0, vmax=delta_t)
        cb = axs.colorbar(im, label='$\Delta$t (hrs.)', loc='b')
        axs.add_feature(cfeature.LAND, facecolor='lightgrey')
        axs.coastlines(resolution=resol, color='k')
        axs.format(boundinglat=60)
        
        plt.savefig(directory_path+'/XO_differences_{}_{}_{}hrs_{}m.png'.format(satellite, analysis, delta_t, dist_req),dpi=300)
        '''
        if delta_t == 1: 
            k = 0 
        elif delta_t == 3:
            k = 1 
        elif delta_t == 6:
            k = 2
        elif delta_t == 12:
            k = 3
        elif delta_t == 24:
            k = 4
        elif delta_t == 48: 
            k = 5
        
        if dist_req == 150: 
            MAD_150m[k] = MAD
            N_150m[k] = len(df_XO_total_nonna)
            RMSE_150m[k] = rmse_val
        elif dist_req == 750:
            MAD_750m[k] = MAD
            N_750m[k] = len(df_XO_total_nonna)
            RMSE_750m[k] = rmse_val
        elif dist_req == 1500:
            MAD_1500m[k] = MAD
            N_1500m[k] = len(df_XO_total_nonna)
            RMSE_1500m[k] = rmse_val
        elif dist_req == 3500:
            MAD_3500m[k] = MAD
            N_3500m[k] = len(df_XO_total_nonna)
            RMSE_3500m[k] = rmse_val
        



#%%


resol='110m'
fig,ax = pplt.subplots(refwidth=2.5, ncols=3, nrows=1, sharey=False)
fig.patch.set_facecolor('white')

axs = ax[0]
axs.scatter(delta_t_val, MAD_150m, c='grey', label='r=150m')
axs.scatter(delta_t_val, MAD_750m, c='red', label='r=750m')
axs.scatter(delta_t_val, MAD_1500m, c='blue', label='r=1500m')
axs.scatter(delta_t_val, MAD_3500m, c='green', label='r=3500m')

axs.legend(loc='lr', ncol=1)
axs.format(xlabel='Within $\Delta$t (hrs.)', ylabel = 'median absolute deviation (cm)')

axs = ax[1]
axs.scatter(delta_t_val, N_150m, c='grey', label='r=150m')
axs.scatter(delta_t_val, N_750m, c='red', label='r=750m')
axs.scatter(delta_t_val, N_1500m, c='blue', label='r=1500m')
axs.scatter(delta_t_val, N_3500m, c='green', label='r=3500m')
axs.set_yscale('log')

axs.format(xlabel='Within $\Delta$t (hrs.)', ylabel = 'number of observaitons')

axs = ax[2]
axs.scatter(delta_t_val, RMSE_150m, c='grey', label='r=150m')
axs.scatter(delta_t_val, RMSE_750m, c='red', label='r=750m')
axs.scatter(delta_t_val, RMSE_1500m, c='blue', label='r=1500m')
axs.scatter(delta_t_val, RMSE_3500m, c='green', label='r=3500m')


axs.format(xlabel='Within $\Delta$t (hrs.)', ylabel = 'root-mean-square-error (cm)')

ax[0:1].format(suptitle='CS2 ESA-E crossover analysis: March 2021')
fig.format(abc='(a)', abcloc='ll')

plt.savefig(directory_path+'/XO_MAD_statistics_{}_{}.png'.format(satellite, analysis),dpi=300)

#txt_fig = 'max abs. diff(XOs) = {:.2f} cm \navg. abs. diff(XOs) = {:.2f} cm\nMAD = {:.2f} cm\nRMSE = {:.2f} cm\nN = {}\navg. $\Delta$t = {:.2f} hrs.'.format(np.nanmax(df_XO_total_nonna['abs_diff_A_B'])*100,np.nanmean(df_XO_total_nonna['abs_diff_A_B'])*100, MAD,rmse_val, len(df_XO_total_nonna), np.nanmean(df_XO_total_nonna['time_delta']))
#axs.format(ultitle=txt_fig,title='{}: $\Delta$t = {} hrs., dist. req. = {} m'.format(analysis, delta_t, dist_req))
#axs.format(xlim=[0, delta_t],xlabel='$\Delta$t (hrs)', ylabel = 'abs. diff. between  {} freeboards at XOs (m)'.format(satellite))




