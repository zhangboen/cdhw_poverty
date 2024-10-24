# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:33:23 2023

@author: cenv1021
"""

import xarray as xr
import os
import numpy as np
import pandas as pd
import glob
import sys

def reprojectNC(ds, lons, lats):
    try:
        ds1 = ds.where(ds.lon>=0, drop = True)
        ds2 = ds.where(ds.lon<0, drop = True)
        ds2 = ds2.assign(lon = 360 + ds2.lon.values)
        ds3 = xr.concat([ds1, ds2], dim = 'lon')
        ds3 = ds3.interp(lon = lons, lat = lats, method = 'linear')
    except:
        ds1 = ds.where(ds.longitude>=0, drop = True)
        ds2 = ds.where(ds.longitude<0, drop = True)
        ds2 = ds2.assign(longitude = 360 + ds2.longitude.values)
        ds3 = xr.concat([ds1, ds2], dim = 'longitude')
        ds3 = ds3.interp(longitude = lons, latitude = lats, method = 'linear')
    return ds3

os.chdir('c:/Research/observation_constrain/')

# read max
dsMask = xr.open_dataset('data/grid_cell_025deg_mask_for_ocean.nc') 
lons = dsMask.lon.values
lats = dsMask.lat.values
mask = dsMask.mask.values
dsMask.close()

# read obs pr
dsAnn = []
dsMax = []
for year in range(2001, 2015):
    name = 'd:/Drought_ExtremePRCP/compare_pre_IMERG/pr.3B-DAY.MS.MRG.3IMERG.%d.V06.precipitationCal.1deg.nc'%year
    ds = xr.open_dataset(name)
    dsAnn0 = ds.sum(dim='time')
    dsMax0 = ds.max(dim='time')
    dsAnn.append(dsAnn0)
    dsMax.append(dsMax0)
dsAnn = xr.concat(dsAnn, dim = 'time')
dsMax = xr.concat(dsMax, dim = 'time')

# reproject obs pr
dsAnn = reprojectNC(dsAnn, lons, lats)
dsMax = reprojectNC(dsMax, lons, lats)

# get model output filenames
model = 'ACCESS-CM2'
ens = 'r1i1p1f1'
df_obs = pd.read_csv('result_csv/climate_variable_%s_%s_2001-2014_catchment_gt_1000_cv_rf.csv'%(model,ens))

# connet with DN
ann = []
for i,year in enumerate(np.arange(2001, 2015)):
    ds_sim = dsMax.where(dsMax.time==i,drop=True)
    val = np.flip(ds_sim.precipitationCal.values,0).ravel()
    name1 = 'prMaxIMERG'
    df0 = pd.DataFrame({'DN':df_obs.DN.unique().astype(int),
                        name1:val[df_obs.DN.unique().astype(int)]})
    df0['year'] = year
    ann.append(df0)
ann = pd.concat(ann)
df_obs = pd.merge(df_obs, df0, on = ['DN','year'], how = 'left')

df_obs['biasDis'] = (df_obs['rf_cv'] - df_obs['runoff_weighted'])/df_obs['runoff_weighted'] * 100
df_obs['biasMax'] = (df_obs['ams'] - df_obs['prMaxIMERG'])/df_obs['prMaxIMERG'] * 100
df_obs['biasAnn'] = (df_obs['annsum'] - df_obs['prAnnIMERG'])/df_obs['prAnnIMERG'] * 100

fig, ax = plt.subplots()
obs = df_obs.biasMax.values
sim = df_obs.biasDis.values
ax.scatter(obs, sim, s = 1)
ax.set_yscale('log')
ax.set_xscale('log')
ax.axline((100,100), (1000,1000), color = 'red', ls = (0,(5,5)))
ax.set_xlabel('Relative bias in AMS precipitaiton (%)')
ax.set_ylabel('Relative bias in AMS streamflow (%)')
fig.savefig('AMS rainfall - AMS streamflow.png', dpi = 600)


