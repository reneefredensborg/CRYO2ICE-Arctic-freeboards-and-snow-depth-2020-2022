# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:30:50 2024

@author: rmfha
"""

import glob
import numpy as np
from netCDF4 import Dataset
import proplot as pplt
import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os 
import pandas as pd
from scipy import signal 

#%%


#type_analysis = 'median' # select median, max, mean
'''
analysis_data = ['2020-11',
                 '2020-12',
                 '2021-01',
                 '2021-02',
                 '2021-03',
                 '2021-04',
                 '2021-11',
                 '2021-12',
                 '2022-01',
                 '2022-02',
                 '2022-03',
                 '2022-04']
'''


analysis_data = ['2020-11',
                 '2020-12',
                 '2021-01',
                 '2021-02',
                 '2021-03',
                 '2021-04']


for j in np.arange(0, len(analysis_data)):
    print('Analysis of: {}'.format(analysis_data[j]))
    directory = r'D:\DTU\CRYO2ICE\OSISAF_MR_drift'
    folder_yr = analysis_data[j][0:4]
    folder_month = analysis_data[j][5:7]

    fp_OSISAF =  glob.glob(directory + '/{}/{}/*.nc'.format(folder_yr, folder_month))
    
    N_x = 379 # horizontal direction
    N_y = 559 # vertical direction
    
    u_comb, v_comb, unc_comb = np.zeros([N_y, N_x, len(fp_OSISAF)])*np.nan,np.zeros([N_y, N_x, len(fp_OSISAF)])*np.nan, np.zeros([N_y, N_x, len(fp_OSISAF)])*np.nan
    drift_comb = np.zeros([N_y, N_x, len(fp_OSISAF)])*np.nan
        
    for i in np.arange(0, len(fp_OSISAF)):
        try: 
            file_in = Dataset(fp_OSISAF[i],"r",format="NETCDF4")
            lat, lon = file_in.variables['lat'][:], file_in.variables['lon'][:]
            u_individual = file_in.variables['dX'][:]
            v_individual = file_in.variables['dY'][:]
            unc_individual = file_in.variables['Uncertainty'][:]
        
            drift = np.sqrt(u_individual**2+v_individual**2)
            mask = mask = ((u_individual==-998) | (v_individual==-998) )
            
            u_individual[mask]=np.nan
            v_individual[mask]=np.nan
            unc_individual[mask]=np.nan
            drift[mask]=np.nan
            
            u_comb[:, :, i] = u_individual
            v_comb[:, :, i] = v_individual
            unc_comb[:, :, i] = unc_individual
            drift_comb[:, :, i] = drift
        except:
            print('File {} not possible to load. Next file.'.format(fp_OSISAF[i]))
    '''
    if type_analysis == 'max':
        
        type_analysis_txt = 'absolute max'
        u_comb_avg = np.nanmax(np.abs(u_comb),axis=2)
        u_comb_avg[u_comb_avg==0]=np.nan
     
        v_comb_avg = np.nanmax(np.abs(v_comb),axis=2)
        v_comb_avg[v_comb_avg==0]=np.nan
        
        unc_comb_avg = np.nanmax(np.abs(unc_comb),axis=2)
        unc_comb_avg[(u_comb_avg==0) | (v_comb_avg==0)] = np.nan
        
        drift_comb_avg = np.nanmax(drift_comb, axis=2)
    elif type_analysis == 'mean':
        
        type_analysis_txt = 'mean'
        u_comb_avg = np.nanmean(u_comb,axis=2)
        u_comb_avg[u_comb_avg==0]=np.nan
        v_comb_avg = np.nanmean(v_comb,axis=2)
        v_comb_avg[v_comb_avg==0]=np.nan
        
        
        unc_comb_avg = np.nanmean(unc_comb,axis=2)
        unc_comb_avg[(u_comb_avg==0) | (v_comb_avg==0)] = np.nan
        drift_comb_avg = np.nanmean(drift_comb, axis=2)
    elif type_analysis == 'median':
        
        type_analysis_txt = 'median'
        u_comb_avg = np.nanmedian(u_comb,axis=2)
        u_comb_avg[u_comb_avg==0]=np.nan
        v_comb_avg = np.nanmedian(v_comb,axis=2)
        v_comb_avg[v_comb_avg==0]=np.nan
        
        
        unc_comb_avg = np.nanmedian(unc_comb,axis=2)
        unc_comb_avg[(u_comb_avg==0) | (v_comb_avg==0)] = np.nan
        drift_comb_avg = np.nanmedian(drift_comb, axis=2)
    else:
        print('Error! Check input to type analysis.')
    
    #drift_comb_avg = np.sqrt(u_comb_avg**2+v_comb_avg**2)
    
    unc_comb_avg[unc_comb_avg == 0]=np.nan
    unc_avg_sum = unc_comb_avg*2
    unc_avg_sum[unc_avg_sum==0]=np.nan
    '''
    
    ## Create dataframe for CRYO2ICE analysis
    
    df_OSISAF = pd.DataFrame({'lat':lat.flatten(), 'lon':lon.flatten()})
    
    # Create OSI SAF figures
    fig = pplt.figure(refwidth=2.5)
    gs = pplt.GridSpec(ncols=4, nrows=2)
    fig.patch.set_facecolor('white')
    
    resol='110m'
    land = cartopy.feature.NaturalEarthFeature('physical', 'land', \
        scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
        
    # MEAN
    type_analysis = 'mean'
    drift_comb_avg = np.nanmean(drift_comb, axis=2)
    drift_comb_avg_hrly = drift_comb_avg/24
    drift_comb_avg_hrly[drift_comb_avg_hrly==0]=np.nan
    drift_comb_avg_max = drift_comb_avg_hrly*3.1654722844444443 # max time difference observed in 2020-2021 (highest of both years)
    drift_comb_avg_max[drift_comb_avg_max==0]=np.nan
    
    df_OSISAF['OSISAF_mean_drift_hrly'] = drift_comb_avg_hrly.flatten()
    
    ax = fig.add_subplot(gs[0, 0], proj='nplaea') 
    im = ax.scatter(lon.flatten(), lat.flatten(), c=drift_comb_avg_hrly.flatten(), cmap='inferno_r', s=0.5, vmin=0, vmax=1.5)
    cb = ax.colorbar(im, label='{} sea ice drift (km/hr)'.format(type_analysis), loc='b')
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax.coastlines(resolution=resol, color='k')
    ax.format(boundinglat=60, title='avg. {} sea ice\ndrift = {:.2f} km/hr'.format(type_analysis, np.nanmean(drift_comb_avg_hrly.flatten())))
    
    
    
    truncated_drift_max = drift_comb_avg_max.flatten()[~np.isnan(drift_comb_avg_max.flatten())]
    N = len(truncated_drift_max)
    N_impact = len(truncated_drift_max[truncated_drift_max>7])
    N_impact2 = len(truncated_drift_max[truncated_drift_max>3.5])
    
    drift_comb_avg_min = drift_comb_avg_hrly*1.5726920966666669 # max time difference observed in 2020-2021 (highest of both years)
    drift_comb_avg_min[drift_comb_avg_max==0]=np.nan
    
    truncated_drift_min = drift_comb_avg_min.flatten()[~np.isnan(drift_comb_avg_min.flatten())]
    N = len(truncated_drift_min)
    N_impact3 = len(truncated_drift_min[truncated_drift_min>7])
    N_impact4 = len(truncated_drift_min[truncated_drift_min>3.5])
    
    drift_comb_avg_mean = drift_comb_avg_hrly*2.3345895377476262 # max time difference observed in 2020-2021 (highest of both years)
    drift_comb_avg_mean[drift_comb_avg_mean==0]=np.nan
    
    truncated_drift_mean = drift_comb_avg_mean.flatten()[~np.isnan(drift_comb_avg_mean.flatten())]
    N = len(truncated_drift_mean)
    N_impact5 = len(truncated_drift_mean[truncated_drift_mean>7])
    N_impact6 = len(truncated_drift_mean[truncated_drift_mean>3.5])
    
    ax4 = fig.add_subplot(gs[1,0])  
    ab1 = ax4.hist(drift_comb_avg_max.flatten(),alpha=0.5, c='grey', histtype='stepfilled', bins=20)
    ab2 = ax4.axvline(np.nanmean(drift_comb_avg_max.flatten()), c='grey', linestyle='--')
    
    ab3 = ax4.hist(drift_comb_avg_min.flatten(),alpha=0.5, c='red', histtype='stepfilled', bins=20)
    ab4 = ax4.axvline(np.nanmean(drift_comb_avg_min.flatten()), c='red', linestyle='--')
    
    ab5 = ax4.hist(drift_comb_avg_mean.flatten(),alpha=0.5, c='blue', histtype='stepfilled', bins=20)
    ab6 = ax4.axvline(np.nanmean(drift_comb_avg_mean.flatten()), c='blue', linestyle='--')
    
    
    ax4.format(title='mean estimates',xlabel='sea ice drift (km) for C2I$\Delta$t using {} estimates'.format(type_analysis), ylabel='frequency (counts per bin)', ultitle='max(C2I$\Delta$t) > 7 km = {:.2f} %\nmax(C2I$\Delta$t) > 3.5 km = {:.2f} %\nmin(C2I$\Delta$t) > 7 km = {:.2f} %\nmin(C2I$\Delta$t) > 3.5 km = {:.2f} %\nmean(C2I$\Delta$t) > 7 km = {:.2f} %\nmean(C2I$\Delta$t) > 3.5 km = {:.2f} %'.format((N_impact/N)*100,(N_impact2/N)*100,(N_impact3/N)*100,(N_impact4/N)*100,(N_impact5/N)*100,(N_impact6/N)*100))
    ax4.set_xlabel('sea ice drift (km) for C2I$\Delta$t')
    
    # MEDIAN
    type_analysis = 'median'
    drift_comb_avg = np.nanmedian(drift_comb, axis=2)
    drift_comb_avg_hrly = drift_comb_avg/24
    drift_comb_avg_hrly[drift_comb_avg_hrly==0]=np.nan
    drift_comb_avg_max = drift_comb_avg_hrly*3.1654722844444443 # max time difference observed in 2020-2021 (highest of both years)
    drift_comb_avg_max[drift_comb_avg_max==0]=np.nan
    
    df_OSISAF['OSISAF_median_drift_hrly'] = drift_comb_avg_hrly.flatten()
    
    ax = fig.add_subplot(gs[0, 1], proj='nplaea') 
    im = ax.scatter(lon.flatten(), lat.flatten(), c=drift_comb_avg_hrly.flatten(), cmap='inferno_r', s=0.5, vmin=0, vmax=1.5)
    cb = ax.colorbar(im, label='{} sea ice drift (km/hr)'.format(type_analysis), loc='b')
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax.coastlines(resolution=resol, color='k')
    ax.format(boundinglat=60, title='avg. {} sea ice\ndrift = {:.2f} km/hr'.format(type_analysis, np.nanmean(drift_comb_avg_hrly.flatten())))
    
    
    truncated_drift_max = drift_comb_avg_max.flatten()[~np.isnan(drift_comb_avg_max.flatten())]
    N = len(truncated_drift_max)
    N_impact = len(truncated_drift_max[truncated_drift_max>7])
    N_impact2 = len(truncated_drift_max[truncated_drift_max>3.5])
    
    drift_comb_avg_min = drift_comb_avg_hrly*1.5726920966666669 # max time difference observed in 2020-2021 (highest of both years)
    drift_comb_avg_min[drift_comb_avg_max==0]=np.nan
    
    truncated_drift_min = drift_comb_avg_min.flatten()[~np.isnan(drift_comb_avg_min.flatten())]
    N = len(truncated_drift_min)
    N_impact3 = len(truncated_drift_min[truncated_drift_min>7])
    N_impact4 = len(truncated_drift_min[truncated_drift_min>3.5])
    
    drift_comb_avg_mean = drift_comb_avg_hrly*2.3345895377476262 # max time difference observed in 2020-2021 (highest of both years)
    drift_comb_avg_mean[drift_comb_avg_mean==0]=np.nan
    
    truncated_drift_mean = drift_comb_avg_mean.flatten()[~np.isnan(drift_comb_avg_mean.flatten())]
    N = len(truncated_drift_mean)
    N_impact5 = len(truncated_drift_mean[truncated_drift_mean>7])
    N_impact6 = len(truncated_drift_mean[truncated_drift_mean>3.5])
    
    ax4 = fig.add_subplot(gs[1,1])  
    ab1 = ax4.hist(drift_comb_avg_max.flatten(),alpha=0.5, c='grey', histtype='stepfilled', bins=20)
    ab2 = ax4.axvline(np.nanmean(drift_comb_avg_max.flatten()), c='grey', linestyle='--')
    
    ab3 = ax4.hist(drift_comb_avg_min.flatten(),alpha=0.5, c='red', histtype='stepfilled', bins=20)
    ab4 = ax4.axvline(np.nanmean(drift_comb_avg_min.flatten()), c='red', linestyle='--')
    
    ab5 = ax4.hist(drift_comb_avg_mean.flatten(),alpha=0.5, c='blue', histtype='stepfilled', bins=20)
    ab6 = ax4.axvline(np.nanmean(drift_comb_avg_mean.flatten()), c='blue', linestyle='--')
    
    
    ax4.format(title='median estimates',xlabel='sea ice drift (km) for C2I$\Delta$t using {} estimates'.format(type_analysis), ylabel='frequency (counts per bin)', ultitle='max(C2I$\Delta$t) > 7 km = {:.2f} %\nmax(C2I$\Delta$t) > 3.5 km = {:.2f} %\nmin(C2I$\Delta$t) > 7 km = {:.2f} %\nmin(C2I$\Delta$t) > 3.5 km = {:.2f} %\nmean(C2I$\Delta$t) > 7 km = {:.2f} %\nmean(C2I$\Delta$t) > 3.5 km = {:.2f} %'.format((N_impact/N)*100,(N_impact2/N)*100,(N_impact3/N)*100,(N_impact4/N)*100,(N_impact5/N)*100,(N_impact6/N)*100))
    ax4.set_xlabel('sea ice drift (km) for C2I$\Delta$t')
    
    # 75th percentile
    type_analysis = '75th perc.'
    drift_comb_avg = np.nanquantile(drift_comb, q=0.75, axis=2)
    drift_comb_avg_hrly = drift_comb_avg/24
    drift_comb_avg_hrly[drift_comb_avg_hrly==0]=np.nan
    drift_comb_avg_max = drift_comb_avg_hrly*3.1654722844444443 # max time difference observed in 2020-2021 (highest of both years)
    drift_comb_avg_max[drift_comb_avg_max==0]=np.nan
    
    df_OSISAF['OSISAF_75thperc_drift_hrly'] = drift_comb_avg_hrly.flatten()

    ax = fig.add_subplot(gs[0, 2], proj='nplaea') 
    im = ax.scatter(lon.flatten(), lat.flatten(), c=drift_comb_avg_hrly.flatten(), cmap='inferno_r', s=0.5, vmin=0, vmax=1.5)
    cb = ax.colorbar(im, label='{} sea ice drift (km/hr)'.format(type_analysis), loc='b')
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax.coastlines(resolution=resol, color='k')
    ax.format(boundinglat=60, title='avg. {} sea ice\ndrift = {:.2f} km/hr'.format(type_analysis, np.nanmean(drift_comb_avg_hrly.flatten())))
    
    
    truncated_drift_max = drift_comb_avg_max.flatten()[~np.isnan(drift_comb_avg_max.flatten())]
    N = len(truncated_drift_max)
    N_impact = len(truncated_drift_max[truncated_drift_max>7])
    N_impact2 = len(truncated_drift_max[truncated_drift_max>3.5])
    
    drift_comb_avg_min = drift_comb_avg_hrly*1.5726920966666669 # max time difference observed in 2020-2021 (highest of both years)
    drift_comb_avg_min[drift_comb_avg_max==0]=np.nan
    
    truncated_drift_min = drift_comb_avg_min.flatten()[~np.isnan(drift_comb_avg_min.flatten())]
    N = len(truncated_drift_min)
    N_impact3 = len(truncated_drift_min[truncated_drift_min>7])
    N_impact4 = len(truncated_drift_min[truncated_drift_min>3.5])
    
    drift_comb_avg_mean = drift_comb_avg_hrly*2.3345895377476262 # max time difference observed in 2020-2021 (highest of both years)
    drift_comb_avg_mean[drift_comb_avg_mean==0]=np.nan
    
    truncated_drift_mean = drift_comb_avg_mean.flatten()[~np.isnan(drift_comb_avg_mean.flatten())]
    N = len(truncated_drift_mean)
    N_impact5 = len(truncated_drift_mean[truncated_drift_mean>7])
    N_impact6 = len(truncated_drift_mean[truncated_drift_mean>3.5])
    
    ax4 = fig.add_subplot(gs[1,2])  
    ab1 = ax4.hist(drift_comb_avg_max.flatten(),alpha=0.5, c='grey', histtype='stepfilled', bins=20)
    ab2 = ax4.axvline(np.nanmean(drift_comb_avg_max.flatten()), c='grey', linestyle='--')
    
    ab3 = ax4.hist(drift_comb_avg_min.flatten(),alpha=0.5, c='red', histtype='stepfilled', bins=20)
    ab4 = ax4.axvline(np.nanmean(drift_comb_avg_min.flatten()), c='red', linestyle='--')
    
    ab5 = ax4.hist(drift_comb_avg_mean.flatten(),alpha=0.5, c='blue', histtype='stepfilled', bins=20)
    ab6 = ax4.axvline(np.nanmean(drift_comb_avg_mean.flatten()), c='blue', linestyle='--')
    
    
    ax4.format(title='75th perc. estimates',xlabel='sea ice drift (km) for C2I$\Delta$t using {} estimates'.format(type_analysis), ylabel='frequency (counts per bin)', ultitle='max(C2I$\Delta$t) > 7 km = {:.2f} %\nmax(C2I$\Delta$t) > 3.5 km = {:.2f} %\nmin(C2I$\Delta$t) > 7 km = {:.2f} %\nmin(C2I$\Delta$t) > 3.5 km = {:.2f} %\nmean(C2I$\Delta$t) > 7 km = {:.2f} %\nmean(C2I$\Delta$t) > 3.5 km = {:.2f} %'.format((N_impact/N)*100,(N_impact2/N)*100,(N_impact3/N)*100,(N_impact4/N)*100,(N_impact5/N)*100,(N_impact6/N)*100))
    ax4.set_xlabel('sea ice drift (km) for C2I$\Delta$t')
    
    # MAX
    type_analysis = 'max'
    drift_comb_avg = np.nanmax(drift_comb, axis=2)
    drift_comb_avg_hrly = drift_comb_avg/24
    drift_comb_avg_hrly[drift_comb_avg_hrly==0]=np.nan
    drift_comb_avg_max = drift_comb_avg_hrly*3.1654722844444443 # max time difference observed in 2020-2021 (highest of both years)
    drift_comb_avg_max[drift_comb_avg_max==0]=np.nan
    
    df_OSISAF['OSISAF_max_drift_hrly'] = drift_comb_avg_hrly.flatten()

    ax = fig.add_subplot(gs[0, 3], proj='nplaea') 
    im = ax.scatter(lon.flatten(), lat.flatten(), c=drift_comb_avg_hrly.flatten(), cmap='inferno_r', s=0.5, vmin=0, vmax=1.5)
    cb = ax.colorbar(im, label='{} sea ice drift (km/hr)'.format(type_analysis), loc='b')
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax.coastlines(resolution=resol, color='k')
    ax.format(boundinglat=60, title='avg. {} sea ice\ndrift = {:.2f} km/hr'.format(type_analysis, np.nanmean(drift_comb_avg_hrly.flatten())))
    
    truncated_drift_max = drift_comb_avg_max.flatten()[~np.isnan(drift_comb_avg_max.flatten())]
    N = len(truncated_drift_max)
    N_impact = len(truncated_drift_max[truncated_drift_max>7])
    N_impact2 = len(truncated_drift_max[truncated_drift_max>3.5])
    
    drift_comb_avg_min = drift_comb_avg_hrly*1.5726920966666669 # max time difference observed in 2020-2021 (highest of both years)
    drift_comb_avg_min[drift_comb_avg_max==0]=np.nan
    
    truncated_drift_min = drift_comb_avg_min.flatten()[~np.isnan(drift_comb_avg_min.flatten())]
    N = len(truncated_drift_min)
    N_impact3 = len(truncated_drift_min[truncated_drift_min>7])
    N_impact4 = len(truncated_drift_min[truncated_drift_min>3.5])
    
    drift_comb_avg_mean = drift_comb_avg_hrly*2.3345895377476262 # max time difference observed in 2020-2021 (highest of both years)
    drift_comb_avg_mean[drift_comb_avg_mean==0]=np.nan
    
    truncated_drift_mean = drift_comb_avg_mean.flatten()[~np.isnan(drift_comb_avg_mean.flatten())]
    N = len(truncated_drift_mean)
    N_impact5 = len(truncated_drift_mean[truncated_drift_mean>7])
    N_impact6 = len(truncated_drift_mean[truncated_drift_mean>3.5])
    
    ax4 = fig.add_subplot(gs[1,3])  
    ab1 = ax4.hist(drift_comb_avg_max.flatten(),alpha=0.5, c='grey', histtype='stepfilled', bins=20)
    ab2 = ax4.axvline(np.nanmean(drift_comb_avg_max.flatten()), c='grey', linestyle='--')
    
    ab3 = ax4.hist(drift_comb_avg_min.flatten(),alpha=0.5, c='red', histtype='stepfilled', bins=20)
    ab4 = ax4.axvline(np.nanmean(drift_comb_avg_min.flatten()), c='red', linestyle='--')
    
    ab5 = ax4.hist(drift_comb_avg_mean.flatten(),alpha=0.5, c='blue', histtype='stepfilled', bins=20)
    ab6 = ax4.axvline(np.nanmean(drift_comb_avg_mean.flatten()), c='blue', linestyle='--')
    
    
    ax4.format(title='max estimates',xlabel='sea ice drift (km) for C2I$\Delta$t using {} estimates'.format(type_analysis), ylabel='frequency (counts per bin)', ultitle='max(C2I$\Delta$t) > 7 km = {:.2f} %\nmax(C2I$\Delta$t) > 3.5 km = {:.2f} %\nmin(C2I$\Delta$t) > 7 km = {:.2f} %\nmin(C2I$\Delta$t) > 3.5 km = {:.2f} %\nmean(C2I$\Delta$t) > 7 km = {:.2f} %\nmean(C2I$\Delta$t) > 3.5 km = {:.2f} %'.format((N_impact/N)*100,(N_impact2/N)*100,(N_impact3/N)*100,(N_impact4/N)*100,(N_impact5/N)*100,(N_impact6/N)*100))
    ax4.set_xlabel('sea ice drift (km) for C2I$\Delta$t')
    
    
    fig.legend([ab1, ab2, ab3, ab4, ab5, ab6], ['drift w/ max(C2I$\Delta$t) = 3.17 hr', 'avg. drift w/ max(C2I$\Delta$t) = 3.17 hr', 'drift w/ min(C2I$\Delta$t) = 1.57 hr', 'avg. drift w/ min(C2I$\Delta$t) = 1.57 hr','drift w/ mean(C2I$\Delta$t) = 2.33 hr', 'avg. drift w/ mean(C2I$\Delta$t) = 2.33 hr'], loc='b', ncols=2)  
    fig.format(suptitle='{}-{}'.format(folder_yr,folder_month), abc='(a)', abcloc='ul')
    
    plt.savefig(directory+'/OSISAF_C2I_sea_ice_drift_{}_{}.png'.format(folder_yr, folder_month),dpi=300)
    
    print('Map-figure created. CRYO2ICE analysis next...')
    
    cryo2ice_2020_2021 = pd.read_csv('D:\DTU\CRYO2ICE\cryo2ice_tracks_v2.csv')

    print ('Identify relevant CRYO2ICE tracks ... ')

    directory = r'D:\DTU\CRYO2ICE\Data\CRYO2ICE_original_MSS\BaselineE'
    files_check_CS2_2020_2021 =  glob.glob(directory + '/2020-2021/'+'*')
    #df_2020_2021 = load_CRYO2ICE_month(files_check_CS2_2020_2021)
    files_check_drift, id_file = [], []
    for i in np.arange(0, len(files_check_CS2_2020_2021)):
        basename_without_ext = os.path.splitext(os.path.basename(files_check_CS2_2020_2021[i]))[0]
        if basename_without_ext[37:43] == (folder_yr+folder_month):
            files_check_drift.append(files_check_CS2_2020_2021[i])
            id_file.append(i)

    print('Load CRYO2ICE data...')

    def load_CRYO2ICE_month(files_check_CS2_data):
        
        def smooth(y, box_pts):
            box = np.ones(box_pts)/box_pts
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth


        li = []
        k = 0
        for filename in files_check_CS2_data:
            df = pd.read_csv(filename, index_col=None, header=0)
            df['ID']=k

            li.append(df)
            k = k+1
            li.append(df)

        frame2 = pd.concat(li, axis=0, ignore_index=True)
        
        return frame2

    CRYO2ICE = load_CRYO2ICE_month(files_check_drift)

    j = 0
    CRYO2ICE['time_delta'] = np.nan
    for k in CRYO2ICE['ID']:
        CRYO2ICE['time_delta'][j]=cryo2ice_2020_2021['time_delta'][k]
        j = j+1

    print('Finding nearest neighbour OSI SAF...')

    def CRYO2ICE_NN(df3_short,df_AMSR2, label1, label2, label3, label4):
        from sklearn.neighbors import BallTree
        import numpy as np
        import pandas as pd

        ## Only data within 825 m of CryoSat-2 footprint (find approximate area of coincident data)
        query_lats = df3_short[['lat']].to_numpy()
        query_lons = df3_short[['lon']].to_numpy()

        tree = BallTree(np.deg2rad(df_AMSR2[['lat', 'lon']].values), leaf_size =15, metric='haversine')

        distances, indices = tree.query(np.deg2rad(np.c_[query_lats, query_lons]), k=1)
        
        data_check = df3_short
        NN_label1, NN_label2, NN_label3, NN_label4 = [], [], [], []
        for i in indices:
            NN_label1 = np.append(NN_label1, df_AMSR2[label1][int(i)])
            NN_label2 = np.append(NN_label2, df_AMSR2[label2][int(i)])
            NN_label3 = np.append(NN_label3, df_AMSR2[label3][int(i)])
            NN_label4 = np.append(NN_label4, df_AMSR2[label4][int(i)])

        data_check[label1] = NN_label1
        data_check[label2] = NN_label2
        data_check[label3] = NN_label3
        data_check[label4] = NN_label4
        
        data_check['OSISAF_mean']=data_check[label1]*data_check['time_delta']
        data_check['OSISAF_median']=data_check[label2]*data_check['time_delta']
        data_check['OSISAF_75th']=data_check[label3]*data_check['time_delta']
        data_check['OSISAF_max']=data_check[label4]*data_check['time_delta']

        
        return data_check

    CRYO2ICE = CRYO2ICE_NN(CRYO2ICE, df_OSISAF, 'OSISAF_mean_drift_hrly','OSISAF_median_drift_hrly','OSISAF_75thperc_drift_hrly','OSISAF_max_drift_hrly')



    # Create OSI SAF figures
    fig,ax = pplt.subplots(refwidth=2.5, ncols=1, nrows=1)

    perc_use = (len(CRYO2ICE['OSISAF_mean'][CRYO2ICE['OSISAF_mean']>3.5])/len(CRYO2ICE))
    ax.hist(CRYO2ICE['OSISAF_mean'],alpha=0.5, c='grey', histtype='stepfilled', bins=20, label='mean > 3.5 km = {:.2f}%'.format(perc_use*100))
    ax.axvline(np.nanmean(CRYO2ICE['OSISAF_mean']), c='grey', linestyle='--')

    perc_use = (len(CRYO2ICE['OSISAF_median'][CRYO2ICE['OSISAF_median']>3.5])/len(CRYO2ICE))
    ax.hist(CRYO2ICE['OSISAF_median'],alpha=0.5, c='red', histtype='stepfilled', bins=20, label='median > 3.5 km = {:.2f}%'.format(perc_use*100))
    ax.axvline(np.nanmean(CRYO2ICE['OSISAF_median']), c='red', linestyle='--')

    perc_use = (len(CRYO2ICE['OSISAF_75th'][CRYO2ICE['OSISAF_75th']>3.5])/len(CRYO2ICE))
    ax.hist(CRYO2ICE['OSISAF_75th'],alpha=0.5, c='blue', histtype='stepfilled', bins=20, label='75th perc. > 3.5 km = {:.2f}%'.format(perc_use*100))
    ax.axvline(np.nanmean(CRYO2ICE['OSISAF_75th']), c='blue', linestyle='--')

    perc_use = (len(CRYO2ICE['OSISAF_max'][CRYO2ICE['OSISAF_max']>3.5])/len(CRYO2ICE))
    ax.hist(CRYO2ICE['OSISAF_max'],alpha=0.5, c='green', histtype='stepfilled', bins=20, label='max > 3.5 km = {:.2f}%'.format(perc_use*100))
    ax.axvline(np.nanmean(CRYO2ICE['OSISAF_max']), c='green', linestyle='--')

    ax.format(xlabel='OSI SAF sea ice drift (km per C2I$\Delta$t per track)', ultitle='{}-{}'.format(folder_yr, folder_month))
    ax.legend(loc='t', ncols=1)

    plt.savefig(directory+'/OSISAF_C2I_sea_ice_drift_CRYO2ICE_{}_{}.png'.format(folder_yr, folder_month),dpi=300)
print('Analysis ended.')

#%% Individual CRYO2ICE point OSI SAF drift with closest OSI SAF product in time to calculate drift for 
import datetime

directory = r'D:\DTU\CRYO2ICE\Data\CRYO2ICE_original_MSS\BaselineE'
files_check_C2I=  glob.glob(directory + '/2020-2021/'+'*')
cryo2ice_2020_2021 = pd.read_csv('D:\DTU\CRYO2ICE\cryo2ice_tracks_v2.csv')
#df_2020_2021 = load_CRYO2ICE_month(files_check_CS2_2020_2021)
frames = []
for i in np.arange(0, len(files_check_C2I)):
#for i in np.arange(0, 1):
    CRYO2ICE_individual = pd.read_csv(files_check_C2I[i], index_col=None, header=0)
    CRYO2ICE_individual['time_delta'] = cryo2ice_2020_2021['time_delta'][i]


    basename_without_ext = os.path.splitext(os.path.basename(files_check_C2I[i]))[0]
    
    print('Load CRYO2ICE data... File {}/{}: {}'.format(i, len(files_check_C2I), basename_without_ext))
    
    C2I_time=datetime.datetime.strptime(basename_without_ext[37:52],"%Y%m%dT%H%M%S")
    
    print('... Find relevant OSI SAF data...')
    folder_yr = basename_without_ext[37:41]
    folder_month = basename_without_ext[41:43]
    
    dir_OSISAF = r'D:\DTU\CRYO2ICE\OSISAF_MR_drift'
    fp_OSISAF =  glob.glob(dir_OSISAF + '/{}/{}/*.nc'.format(folder_yr, folder_month))
    
    N_x = 379 # horizontal direction
    N_y = 559 # vertical direction
    
    delta_time_diff = []
    for j in np.arange(0, len(fp_OSISAF)):
        basename_without_ext_OSISAF = os.path.splitext(os.path.basename(fp_OSISAF[j]))[0]
        OSISAF_time = datetime.datetime.strptime(basename_without_ext_OSISAF[49:61], "%Y%m%d%H%M")
        delta_time_diff.append(np.abs((C2I_time-OSISAF_time).total_seconds()))
    
    idx = np.argmin(delta_time_diff)
    print('... Following OSISAF file selected: {}'.format(fp_OSISAF[idx]))
    file_in = Dataset(fp_OSISAF[idx],"r",format="NETCDF4")
    lat, lon = file_in.variables['lat'][:], file_in.variables['lon'][:]
    u_individual = file_in.variables['dX'][:]
    v_individual = file_in.variables['dY'][:]
    unc_individual = file_in.variables['Uncertainty'][:]
    
    mask = ((u_individual==-998) | (v_individual==-998) )
    
    u_individual[mask] = np.nan
    v_individual[mask] = np.nan
    unc_individual[mask] = np.nan
    
    drift = np.sqrt(u_individual**2+v_individual**2)
    
    unc_individual=unc_individual/1000
    unc_frac_drift = ((unc_individual*2)/np.abs(drift))*100
    unc_frac_u = ((unc_individual)/np.abs(u_individual))*100
    unc_frac_v = ((unc_individual)/np.abs(v_individual))*100
    unc_sum = unc_individual+unc_individual
    
    df_OSISAF = pd.DataFrame({'lat':lat.flatten(), 'lon':lon.flatten(), 'u':u_individual.flatten(), 'v':v_individual.flatten(),'drift_overall':drift.flatten(), 'uncertainty_overall':unc_frac_drift.flatten(), 'uncertainty_u':unc_frac_u.flatten(), 'uncertainty_v':unc_frac_v.flatten()})
    print('... Nearest neighbouring OSI SAF drift...')
    def CRYO2ICE_NN_individual(df3_short,df_AMSR2, label1, label2, label3, label4, label5, label6):
        from sklearn.neighbors import BallTree
        import numpy as np
        import pandas as pd

        ## Only data within 825 m of CryoSat-2 footprint (find approximate area of coincident data)
        query_lats = df3_short[['lat']].to_numpy()
        query_lons = df3_short[['lon']].to_numpy()

        tree = BallTree(np.deg2rad(df_AMSR2[['lat', 'lon']].values), leaf_size =15, metric='haversine')

        distances, indices = tree.query(np.deg2rad(np.c_[query_lats, query_lons]), k=1)
        
        data_check = df3_short
        NN_label1, NN_label2, NN_label3, NN_label4, NN_label5, NN_label6 = [], [], [], [], [], []
        for i in indices:
            NN_label1 = np.append(NN_label1, df_AMSR2[label1][int(i)])
            NN_label2 = np.append(NN_label2, df_AMSR2[label2][int(i)])
            NN_label3 = np.append(NN_label3, df_AMSR2[label3][int(i)])
            NN_label4 = np.append(NN_label4, df_AMSR2[label4][int(i)])
            NN_label5 = np.append(NN_label5, df_AMSR2[label5][int(i)])
            NN_label6 = np.append(NN_label6, df_AMSR2[label6][int(i)])

        data_check[label1] = NN_label1/24
        data_check[label2] = NN_label2
        data_check[label3] = NN_label3/24
        data_check[label4] = NN_label4/24
        data_check[label5] = NN_label5
        data_check[label6] = NN_label6

        
        data_check['OSISAF_drift']=data_check[label1]*data_check['time_delta']
        data_check['OSISAF_unc']=data_check[label2]
        data_check['OSISAF_u']=data_check[label3]*data_check['time_delta']
        data_check['OSISAF_v']=data_check[label4]*data_check['time_delta']
        data_check['OSISAF_unc_u']=data_check[label5]
        data_check['OSISAF_unc_v']=data_check[label6]
        
        return data_check
    
    CRYO2ICE = CRYO2ICE_NN_individual(CRYO2ICE_individual, df_OSISAF, 'drift_overall','uncertainty_overall', 'u', 'v', 'uncertainty_u', 'uncertainty_v')
    print('... Nearest neighbouring OSISAF identified for current file. ')
    frames.append(CRYO2ICE)
    
print('All C2I files computed...')
    
CRYO2ICE_comb = pd.concat(frames)
    
#%%

# Create OSI SAF figures
fig,axs = pplt.subplots(refwidth=2.5, ncols=3, nrows=3, sharex=False, sharey=False)
fig.patch.set_facecolor('white')
#CRYO2ICE_comb_remove_zero = CRYO2ICE_comb[CRYO2ICE_comb['OSISAF_drift']>0]

CRYO2ICE_nonan = CRYO2ICE_comb[CRYO2ICE_comb['OSISAF_drift'].notna()]
CRYO2ICE_comb_remove_zero = CRYO2ICE_nonan

ax=axs[0]
perc_use = (len(CRYO2ICE_comb_remove_zero['OSISAF_drift'][CRYO2ICE_comb_remove_zero['OSISAF_drift']>3.5])/len(CRYO2ICE_comb_remove_zero))
ax.hist(CRYO2ICE_comb_remove_zero['OSISAF_drift'],alpha=0.5, c='grey', histtype='stepfilled', bins=20)
ax.axvline(np.nanmean(CRYO2ICE_comb_remove_zero['OSISAF_drift']), c='grey', linestyle='--')

ax.format(xlabel='OSI SAF sea ice drift (km per C2I$\Delta$t per track)', ylabel='counts',ultitle='drift > 3.5 km = {:.2f}%\nmedian drift = {:.2f} km/C2I$\Delta$t'.format(perc_use*100,np.nanmedian(CRYO2ICE_comb_remove_zero['OSISAF_drift'])))

ax=axs[1]
ax.hist(CRYO2ICE_comb_remove_zero['OSISAF_u'],alpha=0.5, c='grey', histtype='stepfilled', bins=20)
ax.axvline(np.nanmedian(CRYO2ICE_comb_remove_zero['OSISAF_u']), c='grey', linestyle='--')
abs_disp = np.nanmedian(np.abs(CRYO2ICE_comb_remove_zero['OSISAF_u']))
ax.format(xlabel='OSI SAF u-displacement (km per C2I$\Delta$t per track)', ultitle='median u-displacement = {:.2f} km/C2I$\Delta$t\nmin u-displacement = {:.2f} km/C2I$\Delta$t\nmax. u-displacement = {:.2f} km/C2I$\Delta$t\nmedian abs. u-disp. = {:.2f} km/C2I$\Delta$t'.format(np.nanmedian(CRYO2ICE_comb_remove_zero['OSISAF_u']),np.nanmin(CRYO2ICE_comb_remove_zero['OSISAF_u']),np.nanmax(CRYO2ICE_comb_remove_zero['OSISAF_u']), abs_disp))


ax=axs[2]
ax.hist(CRYO2ICE_comb_remove_zero['OSISAF_v'],alpha=0.5, c='grey', histtype='stepfilled', bins=20)
ax.axvline(np.nanmedian(CRYO2ICE_comb_remove_zero['OSISAF_v']), c='grey', linestyle='--')
abs_disp = np.nanmedian(np.abs(CRYO2ICE_comb_remove_zero['OSISAF_v']))
ax.format(xlabel='OSI SAF v-displacement (km per C2I$\Delta$t per track)', ultitle='median v-displacement = {:.2f} km/C2I$\Delta$t\nmin v-displacement = {:.2f} km/C2I$\Delta$t\nmax. v-displacement = {:.2f} km/C2I$\Delta$t\nmedian abs. v-disp. = {:.2f} km/C2I$\Delta$t'.format(np.nanmedian(CRYO2ICE_comb_remove_zero['OSISAF_v']),np.nanmin(CRYO2ICE_comb_remove_zero['OSISAF_v']),np.nanmax(CRYO2ICE_comb_remove_zero['OSISAF_v']), abs_disp))


ax=axs[3]
ax.hist(CRYO2ICE_comb_remove_zero['OSISAF_unc'],alpha=0.5, c='grey', histtype='stepfilled', bins=20)
ax.axvline(np.nanmean(CRYO2ICE_comb_remove_zero['OSISAF_unc']), c='grey', linestyle='--')
ax.format(xlabel='OSISAF sea ice drift relative uncertainty (%)', ylabel='counts',ultitle='median relative uncertainty = {:.2f}%\nmin relative uncertainty = {:.2f}%'.format(np.nanmedian(CRYO2ICE_comb_remove_zero['OSISAF_unc']),np.nanmin(CRYO2ICE_comb_remove_zero['OSISAF_unc'])))

ax=axs[4]
ax.hist(CRYO2ICE_comb_remove_zero['OSISAF_unc_u'],alpha=0.5, c='grey', histtype='stepfilled', bins=20)
ax.axvline(np.nanmean(CRYO2ICE_comb_remove_zero['OSISAF_unc_u']), c='grey', linestyle='--')
ax.format(xlabel='OSISAF u-displacement relative uncertainty (%)', ultitle='median relative uncertainty = {:.2f}%\nmin relative uncertainty = {:.2f}%'.format(np.nanmedian(CRYO2ICE_comb_remove_zero['OSISAF_unc_u']),np.nanmin(CRYO2ICE_comb_remove_zero['OSISAF_unc_u'])))

ax=axs[5]
ax.hist(CRYO2ICE_comb_remove_zero['OSISAF_unc_v'],alpha=0.5, c='grey', histtype='stepfilled', bins=20)
ax.axvline(np.nanmean(CRYO2ICE_comb_remove_zero['OSISAF_unc_v']), c='grey', linestyle='--')
ax.format(xlabel='OSISAF v_displacement relative uncertainty (%)', ultitle='median relative uncertainty = {:.2f}%\nmin relative uncertainty = {:.2f}%'.format(np.nanmedian(CRYO2ICE_comb_remove_zero['OSISAF_unc_v']),np.nanmin(CRYO2ICE_comb_remove_zero['OSISAF_unc_v'])))

axs[1].format(suptitle='2020-2021, {:.2f}% data removed due to NaN'.format((1-len(CRYO2ICE_nonan)/len(CRYO2ICE_comb))*100))



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
        



#resol='110m'
#fig,ax = pplt.subplots(refwidth=2.5, ncols=3, nrows=1, sharey=False)
#fig.patch.set_facecolor('white')

ax = axs[6]
ax.scatter(delta_t_val, MAD_150m, c='grey', label='r=150m')
ax.scatter(delta_t_val, MAD_750m, c='red', label='r=750m')
ax.scatter(delta_t_val, MAD_1500m, c='blue', label='r=1500m')
ax.scatter(delta_t_val, MAD_3500m, c='green', label='r=3500m')

ax.legend(loc='lr', ncol=1)
ax.format(xlabel='Within $\Delta$t (hrs.)', ylabel = 'median absolute deviation (cm)')

ax = axs[7]
ax.scatter(delta_t_val, N_150m, c='grey', label='r=150m')
ax.scatter(delta_t_val, N_750m, c='red', label='r=750m')
ax.scatter(delta_t_val, N_1500m, c='blue', label='r=1500m')
ax.scatter(delta_t_val, N_3500m, c='green', label='r=3500m')
ax.set_yscale('log')

ax.format(xlabel='Within $\Delta$t (hrs.)', ylabel = 'number of observaitons')

ax = axs[8]
ax.scatter(delta_t_val, RMSE_150m, c='grey', label='r=150m')
ax.scatter(delta_t_val, RMSE_750m, c='red', label='r=750m')
ax.scatter(delta_t_val, RMSE_1500m, c='blue', label='r=1500m')
ax.scatter(delta_t_val, RMSE_3500m, c='green', label='r=3500m')

ax.format(xlabel='Within $\Delta$t (hrs.)', ylabel = 'root-mean-square-error (cm)')

axs[7].format(toplabels=('','CS2 ESA-E crossover analysis: March 2021',''))

fig.format(abc='(a)', abcloc='ll')
plt.savefig(dir_OSISAF+'/OSISAF_C2I_sea_ice_drift_CRYO2ICE_individual_2020_2021.png',dpi=300)


#%%
