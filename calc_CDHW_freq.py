#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 00:16:51 2023

@author: boorn
"""

import pandas as pd
import xarray as xr
import numpy as np
import os,glob
import multiprocessing as mp
import time
from sklearn.cluster import DBSCAN

def transform(da):
    # transform 0-360 to -180+180
    try:
        lons = da.longitude.values
        lats = da.latitude.values
    except:
        lons = da.lon.values
        lats = da.lat.values
    if np.min(lons) < 0:
        return da
    else:
        f = lambda x: ((x+180) % 360) - 180
        newlons1 = f(lons)
        ind = np.argsort(newlons1)
        newlons = newlons1[ind]
        if len(da.shape) == 2:
            da = da[:, ind]
            da = xr.DataArray(
                data=da,
                dims=["lat","lon"],
                coords=dict(
                    lon=(["lon"], newlons),
                    lat=(["lat"], lats),
                ),
            )
        elif len(da.shape) == 3:
            da = da[:,:,ind]
            da = xr.DataArray(
                data=da,
                dims=["time","lat","lon"],
                coords=dict(
                    time=(['time'], da.time.values),
                    lon=(["lon"], newlons),
                    lat=(["lat"], lats),
                ),
            )
        return da

def CDHW(scpdsiName, tmaxName, pD = 10, pHW = 90, dName = 'scpdsi', hName = '2t', n_cpu = 8):
    # read data
    ds_tmax = xr.open_dataset(tmaxName)

    ds_scpdsi = xr.open_dataset(scpdsiName)
    ds_scpdsi = xr.Dataset({'scpdsi':transform(ds_scpdsi[dName])})
    # regrid
    ds_scpdsi = ds_scpdsi.interp(lon = ds_tmax.lon.values, 
                            lat = ds_tmax.lat.values)
    scpdsi = ds_scpdsi[dName].values.reshape((ds_scpdsi[dName].shape[0],-1))

    tmax = ds_tmax[hName].values.reshape((ds_tmax[hName].shape[0],-1))

    # create data mask
    mask = ~np.isnan(ds_scpdsi.isel(time=0).scpdsi.values)

    # mask data and from mm to inches
    scpdsi = scpdsi[:,mask.ravel()]
    tmax = tmax[:,mask.ravel()]
    years = np.unique(ds_tmax.time.dt.year.values)
	
    def f(ids, pHW, pD, return_dict):
        for idx in ids:
            tmax0 = tmax[:,idx]
            scpdsi0 = scpdsi[:,idx]

            # heat wave days
            df0 = pd.DataFrame({'doy':ds_tmax.time.dt.dayofyear.values,
                                'tmax':tmax0})
            p0 = df0.groupby('doy').quantile(pHW/100).reset_index()
            df0 = df0.merge(p0, on = 'doy', how = 'left')
            df0['year'] = ds_tmax.time.dt.year.values
            df0['week'] = ds_tmax.time.dt.isocalendar().week.values
            diff0 = pd.to_datetime(ds_tmax.time.values) - pd.to_datetime(ds_tmax.time.values[0])
            df0['dayidx'] = diff0.days.values
            df0 = df0.loc[df0['tmax_x']>=df0['tmax_y'], :]

            # drought weeks
            df1 = pd.DataFrame({'year':ds_scpdsi.time.dt.year.values,
                                'week':ds_scpdsi.time.dt.isocalendar().week.values,
                                'scpdsi':scpdsi0})
            df1 = df1.loc[df1.scpdsi<=df1.scpdsi.quantile(pD/100),:]
            df1['weekidx'] = df1['year'] * 100 + df1['week']

            # join HW and drought
            df2 = df0.merge(df1, on = ['year','week'], how='outer')

            # cluster CDHW events with minimu separation distance of 4 (eps=3)
            # and duration of coincidence at least 3 days (min_samples=3)
            freq = []
            for a in range(3):
                if a == 0:   # independent heat wave
                    dfs = df2.loc[np.isnan(df2.scpdsi),:].reset_index()
                elif a == 1: # independent drought
                    dfs = df2.loc[np.isnan(df2.tmax_x),:].reset_index()
                else:        # CDHW
                    dfs = df2.loc[(~np.isnan(df2.scpdsi))&(~np.isnan(df2.tmax_x)),:]
                    dfs = dfs.reset_index()
                    
                if dfs.shape[0] == 0:
                    freq = freq + np.zeros_like(years).tolist()
                    continue
                else:
                    if a== 1:
                        cluster0 = DBSCAN(eps=3, min_samples=3).fit(dfs[['weekidx']].values)
                    else:
                        cluster0 = DBSCAN(eps=3, min_samples=3).fit(dfs[['dayidx']].values)
                    dfs['cluster'] = cluster0.labels_
                    dfs = dfs.loc[dfs.cluster!=-1, :]
                    
                    freq0 = dfs.groupby('year').nunique()
                    freq0 = freq0.reindex(years, fill_value=0)['cluster']
                    freq = freq + freq0.values.tolist()
            return_dict[idx] = freq
            print('Finish processing %d'%idx)
            
    manager = mp.Manager()
    return_dict = manager.dict()
    
    num0 = np.sum(mask)
    ids = np.arange(0, num0//n_cpu*n_cpu, num0//n_cpu)
    processes = []
    for i in np.arange(n_cpu):
        if i == n_cpu - 1:
            m = np.arange(ids[-1], num0)
        else:
            m = np.arange(ids[i], ids[i+1])
        pro = mp.Process(target=f, args=(m, pHW, pD, return_dict))
        processes.append(pro)
    for p1 in processes:
        p1.start()
    for p1 in processes:
        p1.join()
    while len(return_dict) < num0:
        print(len(return_dict))
        time.sleep(1) # avoid BrokenPipeError
        
    keys = np.array(return_dict.keys())
    out = np.array(return_dict.values())
    p0 = out[np.argsort(keys)]
    
    out = np.zeros((120, np.prod(ds_tmax[hName].shape[1:]))).astype(np.int8)
    out[:,mask.ravel()] = p0.T
    out = out.reshape((120,)+ds_tmax[hName].shape[1:])
    
    ds = xr.Dataset(
        data_vars=dict(
            hw=(["year", "lat", "lon"], out[:40,:,:]),
            dr=(['year','lat','lon'], out[40:80,:,:]),
            cdhw=(['year','lat','lon'], out[80:,:,:]),
        ),
        coords=dict(
            lon=(["lon"], ds_tmax.lon.values),
            lat=(["lat"], ds_tmax.lat.values),
            year=years,
        ),
        attrs=dict(long_name="Annual frequency of CDHW, heat wave and drought"),
    )
    
    return ds
    
if __name__ == '__main__':
    scpdsiName = '../scPDSI_ERA5_025deg_1981-2020.north.nc'
    tmaxName = 'tmax.1981-2020.05-10.north.nc'
    hName = 'tmax'
    n_cpu = 32
    ds = CDHW(scpdsiName, tmaxName, pD = 10, pHW = 90, hName = hName, n_cpu = n_cpu)
    ds.to_netcdf('../CHDW_Heatwave_Drought_ERA5_CPC_ann_90p_freq_1981-2020.north.05-10.nc')
    ds = None
    del ds

    scpdsiName = '../scPDSI_ERA5_025deg_1981-2020.south.nc'
    tmaxName = 'tmax.1981-2020.11-04.south.nc'
    ds = CDHW(scpdsiName, tmaxName, pD = 10, pHW = 90, hName = hName, n_cpu = n_cpu)
    ds.to_netcdf('../CHDW_Heatwave_Drought_ERA5_CPC_ann_90p_freq_1981-2020.south.11-04.nc')
    ds = None
    del ds
    
    
    
