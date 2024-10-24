#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:19:54 2023

@author: boorn
"""

import geopandas as gpd
import os,glob,re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import xarray as xr
import pymannkendall as mk
import matplotlib as mpl

plt.style.use(['science','no-latex']) # require install SciencePlots
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Arial"],  # specify font here
    "font.size":10}) 

path = '/home/boorn/Seagate/compound_poverty/'

def transform_da(da):
    try:
        lons = da.lon.values
        lats = da.lat.values
    except:
        lons = da.longitude.values
        lats = da.latitude.values
    if np.nanmax(lons) < 200:
        da = da
    else:
        f = lambda x: ((x+180) % 360) - 180
        newlons1 = f(lons)
        ind = np.argsort(newlons1)
        newlons = newlons1[ind]
        try:
            da = da[:, :, ind]
            da = xr.DataArray(
                data=da,
                dims=["year","lat","lon"],
                coords=dict(
                    year=(["year"], da.year.values),
                    lon=(["lon"], newlons),
                    lat=(["lat"], lats),
                ),
            )
        except:
            da = da[:, ind]
            da = xr.DataArray(
                data=da,
                dims=["lat","lon"],
                coords=dict(
                    lon=(["lon"], newlons),
                    lat=(["lat"], lats),
                ),
            )
    return da



#%% read CDHW frequency
os.chdir('/home/boorn/research/compound_poverty/')
path1 = '/home/boorn/Elements/2m_dewpoint_tem_ERA5_daymean_1981-2020/'

dsVPD = xr.open_dataset(path1 + 'vpd_weekmean_1981-2020.nc')



#%% read World Bank income classification in 2020
gdf = gpd.read_file('poverty_data/GSAP2_repair_mannual_dissolve_select.shp')
df = pd.read_excel('poverty_data/OGHIST.xlsx', sheet_name = 2)
df = df[['Unnamed: 0',2020]]
df = df.rename(columns={'Unnamed: 0':'country',2020:'type0'})
gdf['income2020'] = 'NULL'
for c in np.unique(gdf.code.values):
    if (df.country!=c).all():
        continue
    gdf.loc[gdf.code==c,'income2020'] = df.loc[df.country==c,'type0'].values[0]

ds2 = xr.open_dataset('poverty_data/GSAP2_repair_mannual_dissolve_select_idx.nc')
ds2 = ds2.interp(lon = da.lon.values, lat = da.lat.values, method = 'nearest')

dsPop = xr.open_dataset('poverty_data/pop_WorldPop_2020_025deg.nc')
daPop = dsPop.Band1.interp(lon=da.lon.values,lat=da.lat.values)

#%% calc for each region
tau = np.repeat(np.nan, gdf.shape[0])
slope = np.repeat(np.nan, gdf.shape[0])
p = np.repeat(np.nan, gdf.shape[0])
freqC = np.repeat(np.nan, gdf.shape[0])
sig = np.repeat(np.nan, gdf.shape[0])
freqN = np.repeat(np.nan, gdf.shape[0])
popNum = np.repeat(np.nan, gdf.shape[0])

freq4 = {}  # save frequency for 4 types of income countries

for i in range(gdf.shape[0]):
    idx = gdf.iloc[i,:].idx
    income = gdf.iloc[i,:].income2020   # L, H, LM, UM
    
    # clip
    intersects = ds2.idx == idx
    da0 = da.where(intersects==True)
    
    # population number
    pop0 = daPop.where(intersects==True).sum().values
    popNum[i] = pop0
    
    # if number of grid cells is less than 5, then skip it
    ncell = np.sum(~np.isnan(da0.isel(year=0))).values
    if np.sum(intersects) < 5 or ncell < 5:
        continue
    
    # region-sum frequency
    freq0 = da0.sum(dim=['lon','lat']).values 
    try:
        mk0 = mk.original_test(freq0/ncell)
        tau[i] = mk0.Tau
        slope[i] = mk0.slope
        p[i] = mk0.p
    except:
        tau[i] = 0
        slope[i] = 0
        p[i] = 1
    
    # sum of frequency to different income economies
    if income not in freq4.keys():
        freq4[income] = freq0
    else:
        freq4[income] = freq4[income] + freq0
    
    # calc frequency sum and change
    freqN[i] = np.sum(freq0)
    n = len(freq0)//2
    freq1 = np.sum(freq0[-n:])
    freq2 = np.sum(freq0[:n])
    freqC[i] = np.where(freq2==0, freq1, freq1 / freq2)

    # estimate significance of freqC (500 bootstrap resampling)
    random = [np.random.permutation(freq0) for x in range(500)]
    random = [np.sum(x[-n:])/np.sum(x[:n]) for x in random]

    if freqC[i] > 1:
        sig0 = np.nansum(freqC[i] > random) / 500
    else:
        sig0 = np.nansum(freqC[i] < random) / 500
    sig[i] = sig0
    
    print(i)
    
gdf['slope'] = slope
gdf['p'] = p
gdf['tau'] = tau
gdf['freqC'] = np.where(np.isinf(freqC), np.nan, freqC)
gdf['freqN'] = freqN
gdf['sig'] = sig
gdf['popNum'] = popNum

#gdf.to_file('poverty_data/GSAP2_CDHW_MSWEP_freq_chg.shp')

#%% plot 3 subplots
# create point GeoDataFrame to plot significant hatch
lons,lats = np.meshgrid(da.lon.values, da.lat.values)
sig = np.ones_like(lons)
slope = np.zeros_like(lons)
for i, idx in enumerate(gdf.idx.values):
    sig[ds2.idx==idx] = np.where(gdf.p.values[i]<=0.05, 0.01, 0.5)
    slope[ds2.idx==idx] = np.where(gdf.slope.values[i]>0, 1, -1)

def plot(fig, ax, varName, vmin, vmax, vcenter, title=None, cmap='RdBu_r'):
    # plot map
    norm = mpl.colors.TwoSlopeNorm(vcenter, vmin, vmax)
    gdf.plot(column=varName, cmap=cmap, norm = norm, ax = ax, 
              linewidths = 0, ec ='none')
    
    # plot coast line
    country = 'https://raw.githubusercontent.com/Boorn123/MyData/main/ne_110m_admin_0_countries.geojson'
    gdfx = gpd.read_file(country)
    gdfx = gdfx.dissolve()
    gdfx.plot(ax = ax, lw = .1, ec = 'k', fc = 'none')
    
    # add significance hatches
    ax.contourf(lons, lats, sig, levels = [0, 0.05, 1], hatches = ['\\\\',None],
                colors = 'none')
    
    # create colorbar
    width = .2
    height = .03
    cax = ax.inset_axes((.05, .2, width, height))
    cbar = fig.colorbar(ax.collections[0], cax=cax, orientation="horizontal",
                        extend = 'max')
    cax.set_title('Change slope', size = 8)
    cax.tick_params(labelsize=8, width=.3, length=2, pad = 3)
    cbar.outline.set_linewidth(.3)
    cax.xaxis.set_minor_locator(ticker.FixedLocator([-0.02,0.02]))
    cax.xaxis.set_major_locator(ticker.FixedLocator([-0.04,0,0.04]))
    cax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    if title is not None:
        ax.set_title(title, size = 10)
    
    # add lonlat ticks
    ax.set_ylim(-60, None)
    ax.yaxis.set_major_locator(ticker.FixedLocator([-45, -15, 15, 45, 75]))
    ax.yaxis.set_major_formatter(lambda x, pos: '%d$^\circ$N'%x if x > 0 else '%d$^\circ$S'%(-x))
    ax.xaxis.set_major_locator(ticker.FixedLocator([-150, -90, -30, 30, 90, 150]))
    ax.xaxis.set_major_formatter(lambda x, pos: '%d$^\circ$E'%x if x > 0 else '%d$^\circ$W'%(-x))
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(.3)

# map of increase
fig, ax = plt.subplots(figsize=(8, 4))
plot(fig, ax, 'slope', -.06, .06, 0, cmap='RdBu_r')

# lineplot of mk trend
ax2 = ax.inset_axes([0, -1, .45, .8])
ll = []
years = np.arange(1981, 2021)
slope4 = []
for name, c in zip(['L','LM','UM','H'], ['#d7191c','#E97A1D','#1a9641','#0571b0']):
    val = freq4[name]
    val = val / np.mean(val)
    mk1 = mk.original_test(val)
    
    slope4.append(mk1.slope)
    
    idx = np.arange(len(val))
    trend_line1 = idx * mk1.slope + mk1.intercept
    
    ax2.plot(years, val, color=c, linestyle=(0, (5, 10)), lw=.3)
    ax2.plot(years, trend_line1, color=c, lw=1)
for y,slope,name,c in zip([0.9,0.8,0.7,0.6],slope4,['L','LM','UM','H'],
                          ['#d7191c','#E97A1D','#1a9641','#0571b0']):
    ax2.text(0.05, y,  name, color = c, transform = ax2.transAxes, size = 9)
    ax2.text(0.17, y,  'Slope = %.3f***'%slope, color = c, 
             transform = ax2.transAxes, size = 9)
ax2.set_ylim(None, 3)
ax2.set_ylabel('Normalized CDHW frequency', loc = 'top')
ax2.set_xlabel('Year')

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

ax1 = ax.inset_axes([.57, -1, .43, .8])
sns.boxplot(x='group_poor', y="slope", data=gdf1, ax=ax1, showfliers = False, 
            palette = 'Spectral_r', width = .6, linewidth = .2)
ax1.set_xlabel('Ventile of poverty rate at $1.90/day')
ax1.set_ylabel('Change slope in CDHW frequency', y = .45)
ax1.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 19, 3)))
ax1.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(1, 19, 1)))
ticks = ['1st',] + ['%dth'%(i+1) for i in np.arange(3, 19, 3)]
ax1.set_xticklabels(ticks)

y = gdf1.groupby('group_poor')['slope'].mean().values
x = np.arange(len(y))
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
idx = x
trend_line1 = idx * slope + intercept
ax1.plot(idx, trend_line1, color='#B00149', lw=1, ls = '--')
# slope and p-value
ax1.text(.3, .87, 'Correlation = %.3f'%r_value, transform = ax1.transAxes)
ax1.text(.3, .78, '$\mathregular{p}$ = %.3f'%p_value, transform = ax1.transAxes)

fig.text(.07, .82, 'a', weight = 'bold', size = 11)
fig.text(.52, .1, 'c', weight = 'bold', size = 11)
fig.text(.07, .1, 'b', weight = 'bold', size = 11)

fig.savefig('/home/boorn/fig1.png', dpi=1000)

#%% 3.20 and 5.50 boxplot
import seaborn as sns
from scipy import stats
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
for j,name in enumerate(['poor320_ln','poor550_ln']):
    gdf['group_poor'] = np.nan
    poor = np.unique(gdf[name])
    for i,g in enumerate(np.arange(0, 100, 5)):
        thres1 = np.nanpercentile(poor, g)
        thres2 = np.nanpercentile(poor, g+5)
        gdf.loc[(gdf[name]>=thres1)&(gdf[name]<=thres2),'group_poor'] = i
    gdf1 = gdf.loc[~np.isnan(gdf['group_poor']), :]
    
    ax1 = axes[j]
    sns.boxplot(x='group_poor', y="slope", data=gdf1, ax=ax1, showfliers = False, 
                palette = 'Spectral_r', width = .6, linewidth = .2)
    ax1.set_xlabel('Ventile of poverty rate at $%.2f/day'%(float(name[4:7])/100))
    ax1.set_ylabel('Change slope in CDHW frequency', y = .45)
    ax1.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 19, 3)))
    ax1.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(1, 19, 1)))
    ticks = ['1st',] + ['%dth'%(i+1) for i in np.arange(3, 19, 3)]
    ax1.set_xticklabels(ticks)
    
    y = gdf1.groupby('group_poor')['slope'].mean().values
    x = np.arange(len(y))

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    idx = x
    trend_line1 = idx * slope + intercept
    ax1.plot(idx, trend_line1, color='#B00149', lw=1, ls = '--')
    # slope and p-value
    ax1.text(.3, .87, 'Correlation = %.3f'%r_value, transform = ax1.transAxes)
    ax1.text(.3, .78, '$\mathregular{p}$ = %.3f'%p_value, transform = ax1.transAxes)

fig.tight_layout()
fig.savefig('/home/boorn/figS1.png', dpi=1000)

# #%% plot single kdeplot
# fig, ax = plt.subplots()
# g = sns.kdeplot(data=gdf.loc[gdf.income2020.isin(['L','LM','UM','H']),:], 
#                 x = 'slope', hue = 'income2020', ax = ax,
#                 palette=['#0571b0','#1a9641','#E97A1D','#d7191c'])
# g.legend_.set_title(None)
# ax.axvline(x = 0, ls = 'dashed', lw = 1, color = 'k')
# ax.set_xlabel('Normalized change in CDHW frequency')
# fig.savefig('kdeplot_CDHW_4income.pdf')