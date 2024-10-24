#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:19:54 2023

@author: boorn
"""
# module load Anaconda3; module load ecCodes

import geopandas as gpd
import os,glob,re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import xarray as xr
import pymannkendall as mk
import matplotlib as mpl
import scienceplots
from pyogrio import read_dataframe
import matplotlib.font_manager as font_manager
from src.plot_utils import plot_map
import matplotlib as mpl
from src.transform import transform360to180

# set custom font (not a good idea)
path0 = os.environ['DATA']+'/fonts/Helvetica/Helvetica.ttf'
font_manager.fontManager.addfont(path0)
path0 = os.environ['DATA']+'/fonts/Helvetica/Helvetica-Bold.ttf'
font_manager.fontManager.addfont(path0)

# plot style
plt.style.use(['science','no-latex','nature']) # require install SciencePlots
plt.rcParams.update({"font.size":10, 'font.family':'Helvetica'}) 

# set data path
path = '../data/'

# read mask
dsMask = xr.open_dataset(path+'mask.nc')
daMask = xr.DataArray(
    data=dsMask.mask.values,
    dims=["lat","lon"],
    coords=dict(
        lon=(["lon"], dsMask.longitude.values),
        lat=(["lat"], dsMask.latitude.values),
    ),
)
daMask = transform360to180(daMask) * 1

# read calculated CHDW frequency for each GSAP region
gdf = read_dataframe('../poverty_data/GSAP2_CDHW_CHDW_Heatwave_Drought_ERA5_ann_freq_1981-2020_1991-2020.gpkg')

ds2 = xr.open_dataset('../poverty_data/GSAP2_repair_mannual_dissolve_select_idx.nc')
ds2 = ds2.interp(lon = daMask.lon.values, lat = daMask.lat.values, method = 'nearest')

#%% plot
# create point GeoDataFrame to plot significant hatch
lons,lats = np.meshgrid(daMask.lon.values, daMask.lat.values)
sig = np.ones_like(lons)
slope = np.zeros_like(lons)
for i, idx in enumerate(gdf.idx.values):
    sig[ds2.idx==idx] = np.where(gdf.p.values[i]<=0.05, 0.01, 0.5)
    slope[ds2.idx==idx] = np.where(gdf.slope.values[i]>0, 1, -1)

# read averaged cdhw frequency for 4 income groups
freq4 = pd.read_csv('../poverty_data/CDHW_ann_4income_CHDW_Heatwave_Drought_ERA5_ann_freq_1981-2020_1991-2020.csv')

# map of increase
fig, ax = plt.subplots(figsize=(8, 4.5))
plot_map(fig, 
        ax, 
        gdf, 
        'slope', 
        vmin = -6, 
        vmax = 6, 
        vcenter = 0, 
        lons = lons, 
        lats = lats, 
        sigArr = sig, 
        title = 'Historical CDHW change (1991-2020)',
        label = 'Normalized change slope',
        fontsize = 9,
        cmap='RdBu_r')

# lineplot of mk trend
ax2 = ax.inset_axes([0, -1.05, .35, .9])
ll = []
years = np.arange(1991, 2021)
slope4 = []
for name, c in zip(['L','LM','UM','H'], ['#d7191c','#E97A1D','#1a9641','#0571b0']):
    val = freq4[name]
    val = val / np.mean(val)
    mk1 = mk.original_test(val)
    print(name, mk1.slope, mk1.p)
    
    slope4.append(mk1.slope)
    
    idx = np.arange(len(val))
    trend_line1 = idx * mk1.slope + mk1.intercept
    
    ax2.plot(years, val, color=c, linestyle=(0, (5, 10)), lw=.5)
    ax2.plot(years, trend_line1, color=c, lw=1)
for y,slope,name,c in zip([0.9,0.8],slope4[:2],['LIC','LMIC'], ['#d7191c','#E97A1D']):
    ax2.text(0.05, y,  name + ' Slope = %.3f***'%slope, color = c, transform = ax2.transAxes, size = 9, ha = 'left')
for y,slope,name,c in zip([0.2,0.1],slope4[2:],['UMIC','HIC'], ['#1a9641','#0571b0']):
    ax2.text(0.95, y,  name + ' Slope = %.3f***'%slope, color = c, transform = ax2.transAxes, size = 9, ha = 'right', va = 'top')
ax2.set_ylim(-.4, 2.8)
ax2.set_ylabel('Normalized CDHW frequency', fontsize = 9)
ax2.set_xlabel('Year', fontsize = 8)

# boxplot of poor share vs. freqC
import seaborn as sns
gdf['group_poor'] = np.nan
name = "poor190_ln"
poor = np.unique(gdf[name])
for i,g in enumerate(np.arange(0, 100, 5)):
    thres1 = np.nanpercentile(poor, g)
    thres2 = np.nanpercentile(poor, g+5)
    gdf.loc[(gdf[name]>=thres1)&(gdf[name]<=thres2),'group_poor'] = i
gdf1 = gdf.loc[~np.isnan(gdf['group_poor']), :]

# kdeplot show Bootstrap analysis
ax3 = ax.inset_axes([.44, -1.05, .13, .9])
def bootstrap(freq):
    samples = np.array([np.random.choice(freq[(x*2):(x*2+2)]) for x in range(15)])
    samples = samples / np.mean(samples)
    mk1 = mk.original_test(samples)
    return (mk1.slope/2)
df1 = pd.DataFrame({'LIC':[bootstrap(freq4['L']) for i in range(500)],
                    'HIC':[bootstrap(freq4['H']) for i in range(500)],
                    'idx':np.arange(500)}).melt(id_vars = 'idx', var_name = 'name', value_name = 'slope')
sns.boxplot(data = df1, x = 'name', y = 'slope', hue = 'name', ax = ax3, palette = ['#d7191c','#0571b0'], width = .6)
ax3.set_ylabel('Bootstrap resampling of Sen\'s slope', fontsize = 9)
ax3.set_xlabel(None)

ax1 = ax.inset_axes([.65, -1.05, .35, .9])
sns.boxplot(x='group_poor', y="slope", data=gdf1, ax=ax1, showfliers = False, 
            palette = 'Spectral_r', width = .6, linewidth = .4, hue = 'group_poor', legend = False)
ax1.set_xlabel('Ventile of poverty rate at $1.90/day', fontsize = 9)
ax1.set_ylabel('Trend of annual CDHW frequency (%)', fontsize = 9)
ax1.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 19, 3)))
ax1.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(1, 19, 1)))
ticks = ['1st',] + ['%dth'%(i+1) for i in np.arange(3, 19, 3)]
ax1.set_xticklabels(ticks)

# linear regression
y = gdf1.groupby('group_poor')['slope'].mean().values
x = np.arange(len(y))
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
idx = x
trend_line1 = idx * slope + intercept
ax1.plot(idx, trend_line1, color='#B00149', lw=1, ls = '--')
# slope and p-value
ax1.text(.3, .87, 'Correlation = %.3f'%r_value, transform = ax1.transAxes, size = 8)
ax1.text(.3, .78, '$\mathregular{p}$ = %.3f'%p_value, transform = ax1.transAxes, size = 8)

fig.text(.07, .76, 'a', weight = 'bold', size = 11)
fig.text(.07, .16, 'b', weight = 'bold', size = 11)
fig.text(.42, .16,  'c', weight = 'bold', size = 11)
fig.text(.58, .16, 'd', weight = 'bold', size = 11)

fig.savefig('../picture/tmp.png', dpi=1000)