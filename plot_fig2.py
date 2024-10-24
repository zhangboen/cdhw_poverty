#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 17:55:31 2023

@author: boorn
"""

import os,glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import seaborn as sns
from matplotlib import ticker

plt.style.use(['science','no-latex']) # require install SciencePlots
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Arial"],  # specify font here
    "font.size":10}) 

path2 = '../data/cmip6_pr_tmax_tmin/'
os.chdir('/data/ouce-drift/cenv1021/compound_poverty/')
dsMask = xr.open_dataset(path2 + 'mask_4income_1.5deg.nc')

#%% read frequency and severity
dsHISTfreq = []
dsNATfreq = []
dsHISTseve = []
dsNATseve = []
for name in glob.glob(path2 + 'HW_DR_CHDW_ann_freq_seve_HIST_NAT*'):
    ds = xr.open_dataset(name)
    ds0 = ds.interp(lon=dsMask.lon.values,lat=dsMask.lat.values,
                    kwargs={"fill_value": "extrapolate"},
                    method = 'nearest')
    dsHISTfreq.append(ds0.freqHISTcdhw)
    dsNATfreq.append(ds0.freqNATcdhw)
    dsHISTseve.append(ds0.seveHISTcdhw)
    dsNATseve.append(ds0.seveNATcdhw)
    ds.close();ds=None
    
dsHISTfreq = xr.concat(dsHISTfreq, dim = 'model')
dsNATfreq = xr.concat(dsNATfreq, dim = 'model')
dsHISTseve = xr.concat(dsHISTseve, dim = 'model')
dsNATseve = xr.concat(dsNATseve, dim = 'model')

dsHISTseve = dsHISTseve.where(dsHISTseve > 0)
dsNATseve = dsNATseve.where(dsNATseve > 0)

#%% calc difference and sig
dsHIST0 = dsHISTfreq.sum(dim='year')
dsNAT0 = dsNATfreq.sum(dim='year')
dsFreq = dsHIST0.mean(dim='model') / dsNAT0.mean(dim='model')

dsHISTs0 = dsHISTseve.where(dsHISTseve > 0).mean(dim='year',skipna=True)
dsNATs0 = dsNATseve.where(dsNATseve > 0).mean(dim='year',skipna=True)
dsSeve = dsHISTs0.mean(dim='model') / dsNATs0.mean(dim='model')

#%% transform grid
lons = dsFreq.lon.values
lats = dsFreq.lat.values
f = lambda x: ((x+180) % 360) - 180
newlons1 = f(lons)
ind = np.argsort(newlons1)
newlons = newlons1[ind]
dsFreq = dsFreq[:, ind]
dsSeve = dsSeve[:, ind]
ds = xr.Dataset(
    data_vars=dict(
        freq=(["lat", "lon"], dsFreq.values),
        seve=(["lat", "lon"], dsSeve.values),
        mask=(['lat','lon'], dsMask.mask[:,ind].values),
    ),
    coords=dict(
        lon=(["lon"], newlons),
        lat=(["lat"], lats),
    ),
)
dsFreq = ds.freq
dsSeve = ds.seve

#%% read poverty
dsPoverty = xr.open_dataset('shp/GSAP2_p190_025deg.nc')
dsPoverty = dsPoverty.interp(lon=ds.lon.values,
                             lat=ds.lat.values,
                             kwargs={"fill_value": "extrapolate"},
                             method = 'nearest')

df = pd.DataFrame({'poverty':dsPoverty.Band1.values.ravel(),
                   'freq':dsFreq.values.ravel(),
                   'seve':dsSeve.values.ravel(),
                   'region':ds.mask.values.ravel()})
df = df.loc[~np.isnan(df.freq),:]
df = df.loc[~np.isnan(df.poverty),:]

#%% plot freq only

# boxplot of poor share vs. freqC
# last group is >= 85
df['group0'] = np.nan
poor = np.unique(df.poverty.values)
for i,g in enumerate(np.arange(0, 100, 5)):
    thres1 = np.nanpercentile(poor, g)
    thres2 = np.nanpercentile(poor, g+5)
    df.loc[(df.poverty>=thres1)&(df.poverty<=thres2),'group0'] = i
df1 = df.loc[~np.isnan(df.group0), :]

fig, axes = plt.subplots(1, 2, figsize = (8, 3))
p = sns.kdeplot(data=df,hue='region',x='freq', cumulative = False, 
                palette = 'tab10', common_norm = False, ax = axes[0],
                clip = [0, 3.5])
axes[0].set_xlabel('Anthropogenic impacts on CDHWs')
axes[0].legend(title=None, loc='upper right',
                 labels=['HIC','UMIC','LMIC','LIC'])
axes[0].set_ylabel('Density')

# plot filled region with x >= 2 and LIC
x3 = p.get_lines()[-1].get_xdata()
y3 = p.get_lines()[-1].get_ydata()
y3 = y3[x3>=2]
x3 = x3[x3>=2]
axes[0].fill_between(x3, np.zeros_like(x3), y3, fc = '#0D1490', alpha = .3)

frac = np.sum((ds.mask==1)&(dsFreq>=2)) / np.sum(ds.mask==1) * 100
axes[0].text(2.6, 0.3, '%d'%frac+'%', size = 11)

# plot filled region with x >= 2 and HIC
y1 = 0
x2 = p.get_lines()[0].get_xdata()
y2 = p.get_lines()[0].get_ydata()
y2 = y2[x2>=2]
x2 = x2[x2>=2]
axes[0].fill_between(x2, np.zeros_like(x2), y2, fc = '#9D2119', alpha = .3)

frac = np.sum((ds.mask==4)&(dsFreq>=2)) / np.sum(ds.mask==4) * 100
axes[0].text(1.55, 0.1, '%.1f'%frac+'%', size = 11)

# plot boxplot and regplot
ax1 = axes[1]
sns.boxplot(x='group0', y="freq", data=df1, ax=ax1, showfliers = False, 
            palette = 'Spectral_r', width = .6, linewidth = .4)
ax1.set_xlabel('Ventile of poverty rate at $1.90/day')
ax1.set_ylabel('Anthropogenic impacts on CDHWs', labelpad = 2, y = .45)
ax1.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 19, 3)))
ax1.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(1, 19, 1)))
ticks = ['1st',] + ['%dth'%(i+1) for i in np.arange(3, 19, 3)]
ax1.set_xticklabels(ticks)
y = df1.groupby('group0')['freq'].mean().values
x = np.arange(y.shape[0])
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
idx = x
trend_line1 = idx * slope + intercept
ax1.plot(idx, trend_line1, color='#B00149', lw=1, ls = '--')
ax1.set_ylim(0.7, 3.4)
# slope and p-value
ax1.text(.1, .85, 'Correlation = %.3f\np = %.3f'%(r_value, p_value), 
         transform = ax1.transAxes)

#fig.tight_layout()

axes[0].text(-.15, 1, 'a', weight='bold', transform = axes[0].transAxes, size = 12)
axes[1].text(-.15, 1, 'b', weight='bold', transform = axes[1].transAxes, size = 12)

fig.savefig('picture/fig2.png', dpi = 1000)

# #%% plot freq and severity

# # boxplot of poor share vs. freqC
# df['group0'] = np.nan
# for i,g in enumerate(np.arange(0, 90, 5)):
#     if g == 85:
#         df.loc[df.poverty>=g,'group0'] = i
#     else:
#         df.loc[(df.poverty>=g)&(df.poverty<=g+5),'group0'] = i
# df1 = df.loc[~np.isnan(df.group0), :]

# fig, axes = plt.subplots(2, 2, figsize = (8, 6))
# sns.kdeplot(data=df,hue='region',x='freq', cumulative = True, 
#             palette = 'tab10', common_norm = False, ax = axes[0,0],
#             clip = [0, 3.4])
# axes[0,0].set_xlabel('$\Delta$ CDHW frequency ratio')
# axes[0,0].legend(title=None, loc='lower right',
#                  labels=['HIC','UMIC','LMIC','LIC'])
# axes[0,0].set_ylabel('Cumulative probability')

# sns.kdeplot(data=df,hue='region',x='seve', cumulative = True, 
#             palette = 'tab10', common_norm = False, ax = axes[0,1],
#             clip = [0, 3.4], legend = False)
# axes[0,1].set_xlabel('$\Delta$ CDHW severity')
# axes[0,1].set_ylabel('Cumulative probability')

# axes[0,0].axvline(x = 2, ls = 'dashed', lw = .3, color = 'k')
# axes[0,0].axhline(y = 0.58, ls = 'dashed', lw = .3, color = 'k', xmin = 0, 
#                   xmax = 2/3.4 - .04)
# axes[0,0].axhline(y = 0.96, ls = 'dashed', lw = .3, color = 'k', xmin = 0, 
#                   xmax = 2/3.4 - .04)

# ax1 = axes[1,0]
# sns.boxplot(x='group0', y="freq", data=df1, ax=ax1, showfliers = False, 
#             palette = 'Spectral_r', width = .6, linewidth = .2)
# ax1.set_xlabel('Share of population that is poor ($5.50/day) (%)', loc = 'right')
# ax1.set_ylabel('CDHW frequency ratio', labelpad = 2)
# ax1.set_xticks(np.arange(1.5, 18, 2))
# ax1.set_xticklabels([int((i+.5)//2*10) for i in np.arange(1.5, 18, 2)])
# ax1.tick_params(which='minor', length=0, axis = 'x')
# y = df1.groupby('group0')['freq'].mean()
# x = np.arange(y.shape[0])
# from scipy import stats
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# idx = np.arange(20)
# trend_line1 = idx * slope + intercept
# ax1.plot(idx, trend_line1, color='#B00149', lw=1, ls = '--')
# # slope and p-value
# ax1.text(.3, .85, 'Slope = %.4f%s'%(slope,'*' if p_value <=0.05 else ''), 
#          transform = ax1.transAxes)

# ax1 = axes[1,1]
# sns.boxplot(x='group0', y="seve", data=df1, ax=ax1, showfliers = False, 
#             palette = 'Spectral_r', width = .6, linewidth = .2)
# ax1.set_xlabel('Share of population that is poor ($5.50/day) (%)', loc = 'right')
# ax1.set_ylabel('CDHW severity ratio', labelpad = 2)
# ax1.set_xticks(np.arange(1.5, 18, 2))
# ax1.set_xticklabels([int((i+.5)//2*10) for i in np.arange(1.5, 18, 2)])
# ax1.tick_params(which='minor', length=0, axis = 'x')
# y = df1.groupby('group0')['seve'].mean()
# x = np.arange(y.shape[0])
# from scipy import stats
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# idx = np.arange(20)
# trend_line1 = idx * slope + intercept
# ax1.plot(idx, trend_line1, color='#B00149', lw=1, ls = '--')
# # slope and p-value
# ax1.text(.3, .85, 'Slope = %.4f%s'%(slope,'*' if p_value <=0.05 else ''), 
#          transform = ax1.transAxes)
# fig.tight_layout()

# #fig.savefig('picture/kdeplot_boxplot_freq_seve_vs_poverty.pdf')

