#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 20:33:21 2023

@author: boorn
"""

import os,glob,re
import numpy as np
import matplotlib.pyplot as plt
import cmaps,string
from matplotlib import ticker
import pandas as pd
import xarray as xr
import pymannkendall as mk
from sklearn.linear_model import LinearRegression
from matplotlib.transforms import Affine2D
import seaborn as sns
import scienceplots

plt.style.use(['science','no-latex']) # require install SciencePlots
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Arial"],  # specify font here
    "font.size":10}) 

#%% read CDHW frequency and severity
path1 = 'd:/compound_poverty/'
path2 = 'd:/cmip6_pr_tmax_tmin/'

os.chdir('g:/research/compound_poverty/')

dfObs = pd.read_csv('poverty_data/Drought_Heat_CDHW_ann_freq_4income.csv')

dsMask = xr.open_dataset(path2 + 'mask_4income_1.5deg.nc')

dfHIST = []
dfNAT = []
for name in glob.glob(path2 + 'HW_DR_CHDW_ann_freq_seve_HIST_NAT*'):
    a = os.path.basename(name).split('_')
    model = a[8]
    ens = a[9]
    ds = xr.open_dataset(name)
    dsMask1 = dsMask.interp(lon=ds.lon.values,lat=ds.lat.values,
                            kwargs={"fill_value": "extrapolate"})
    val1 = ds.where(dsMask1.mask==1).sum(dim=['lon','lat']).freqHISTcdhw.values
    val2 = ds.where(dsMask1.mask==2).sum(dim=['lon','lat']).freqHISTcdhw.values
    val3 = ds.where(dsMask1.mask==3).sum(dim=['lon','lat']).freqHISTcdhw.values
    val4 = ds.where(dsMask1.mask==4).sum(dim=['lon','lat']).freqHISTcdhw.values
    df = pd.DataFrame({
        'L':val1,
        'LM':val2,
        'UM':val3,
        'H':val4,
        'year':np.arange(1981,2021)})
    df['model'] = model
    df['ens'] = ens
    dfHIST.append(df)
    
    val1 = ds.where(dsMask1.mask==1).sum(dim=['lon','lat']).freqNATcdhw.values
    val2 = ds.where(dsMask1.mask==2).sum(dim=['lon','lat']).freqNATcdhw.values
    val3 = ds.where(dsMask1.mask==3).sum(dim=['lon','lat']).freqNATcdhw.values
    val4 = ds.where(dsMask1.mask==4).sum(dim=['lon','lat']).freqNATcdhw.values
    df = pd.DataFrame({
        'L':val1,
        'LM':val2,
        'UM':val3,
        'H':val4,
        'year':np.arange(1981,2021)})
    df['model'] = model
    df['ens'] = ens
    dfNAT.append(df)
    ds.close();ds=None
    
    print(name)
dfHIST = pd.concat(dfHIST)
dfNAT = pd.concat(dfNAT)

for a,i in enumerate(['L','LM','UM','H']):
    dfHIST[i+'_d'] = dfHIST[i] / np.nansum(dsMask1.mask.values==a+1)
    dfNAT[i+'_d'] = dfNAT[i] / np.nansum(dsMask1.mask.values==a+1)
    
dfHIST['LLM'] = dfHIST['L'] + dfHIST['LM']
dfHIST['HUM'] = dfHIST['H'] + dfHIST['UM']
dfNAT['LLM'] = dfNAT['L'] + dfNAT['LM']
dfNAT['HUM'] = dfNAT['H'] + dfNAT['UM']

# Normalize
dfObs = dfObs / dfObs.mean(axis=0)
dfMean = dfHIST.groupby(['model','ens']).mean().reset_index()
dfHIST = dfHIST.merge(dfMean, on = ['model','ens'], suffixes=['','_m'])
dfMean = dfNAT.groupby(['model','ens']).mean().reset_index()
dfNAT = dfNAT.merge(dfMean, on = ['model','ens'], suffixes=['','_m'])
for i in ['L','LM','UM','H','LLM','HUM']:
    dfHIST[i+'_n'] = dfHIST[i].values / dfHIST[i+'_m'].values
    dfNAT[i+'_n'] = dfNAT[i].values / dfNAT[i+'_m'].values

#%% attribution
betaHIST = []
betaNAT = []
attrTrHIST = []
attrTrNAT = []
for name in glob.glob(path2 + 'HW_DR_CHDW_ann_freq_seve_HIST_NAT*'):
    a = os.path.basename(name).split('_')
    model = a[8]
    ens = a[9]
    
    dfHIST0 = dfHIST.loc[(dfHIST.model==model)&(dfHIST.ens==ens),:]
    dfNAT0 = dfNAT.loc[(dfNAT.model==model)&(dfNAT.ens==ens),:]
    
    beta_hist = {}
    beta_nat = {}
    attr_hist = {}
    attr_nat = {}
    for label in ['L','LM','UM','H']:
        hist0 = dfHIST0[label+'_n'].values
        nat0 = dfNAT0[label+'_n'].values
        obs0 = dfObs[label+'DH'].values
        #hist0 = hist0 - nat0
        
        reg_hist0 = LinearRegression().fit(hist0[:,None], obs0)
        beta_hist0 = reg_hist0.coef_
        
        reg_nat0 = LinearRegression().fit(nat0[:,None], obs0)
        beta_nat0 = reg_nat0.coef_
    
        # attr change
        mk0 = mk.original_test(hist0)
        a = mk0.slope * beta_hist0
        attr_hist[label] = a
        
        mk0 = mk.original_test(nat0)
        a = mk0.slope * beta_nat0
        attr_nat[label] = a
    
        beta_hist[label] = beta_hist0
        beta_nat[label] = beta_nat0
    
    df0 = pd.DataFrame.from_dict(beta_hist)
    df0['model'] = model
    df0['ens'] = ens
    betaHIST.append(df0)
    
    df0 = pd.DataFrame.from_dict(beta_nat)
    df0['model'] = model
    df0['ens'] = ens
    betaNAT.append(df0)
    
    df0 = pd.DataFrame.from_dict(attr_hist)
    df0['model'] = model
    df0['ens'] = ens
    attrTrHIST.append(df0)
    
    df0 = pd.DataFrame.from_dict(attr_nat)
    df0['model'] = model
    df0['ens'] = ens
    attrTrNAT.append(df0)

betaHIST = pd.concat(betaHIST).melt(id_vars=['model','ens'])
betaNAT = pd.concat(betaNAT).melt(id_vars=['model','ens'])

attrTrHIST = pd.concat(attrTrHIST).melt(id_vars=['model','ens'])
attrTrNAT = pd.concat(attrTrNAT).melt(id_vars=['model','ens'])

betaHIST['type'] = 'ANT'
betaNAT['type'] = 'NAT'
beta = pd.concat([betaHIST, betaNAT])

attrTrHIST['type'] = 'ANT'
attrTrNAT['type'] = 'NAT'
attr = pd.concat([attrTrHIST, attrTrNAT])

#%% plot trend and attribution
fig, axes = plt.subplots(2, 2, figsize=(8, 5.5))
years = np.arange(1981, 2021)
for ax,label,name in zip([axes[0,0], axes[0,1]], ['L_n','H_n'],['LDH','HDH']):
    OBS = dfObs[name].values
    HISTmean = dfHIST.groupby('year')[label].median().values
    NATmean = dfNAT.groupby('year')[label].median().values

    HISTlow = dfHIST.groupby('year')[label].quantile(.05).values
    HISTupp = dfHIST.groupby('year')[label].quantile(.95).values
    NATlow = dfNAT.groupby('year')[label].quantile(.05).values
    NATupp = dfNAT.groupby('year')[label].quantile(.95).values
    
    l1, = ax.plot(years, OBS, color = 'k', lw=.8, zorder = 4)
    l2, = ax.plot(years, HISTmean, color='#DA4116', lw=.8, zorder = 3)
    l3, = ax.plot(years, NATmean, color='#2455E0', lw=.8, zorder = 2)
    
    ax.fill_between(years, HISTlow, HISTupp, fc = 'red', alpha = .3, zorder = 1)
    ax.fill_between(years, NATlow, NATupp, fc = '#2455E0', alpha = .3, zorder = 0)
    
    ax.set_xlim(years[0], years[-1])
    ax.set_xlabel('Year')
    ax.set_ylabel('Normalized frequency')
    
    mk1 = mk.original_test(OBS)
    mk2 = mk.original_test(HISTmean)
    mk3 = mk.original_test(NATmean)
    
    for mk0,c in zip([mk1,mk2,mk3],['k','#DA4116','#2455E0']):
        idx = np.arange(len(years))
        trend_line1 = idx * mk0.slope + mk0.intercept
        ax.plot(years, trend_line1, color=c, lw=.3, ls = 'dashed')
    
    if name == 'HDH':
        x = .05
    else:
        x = .05
    for y, mk0, c in zip([.92,.83,.75], [mk1,mk2,mk3],['k','#DA4116','#2455E0']):
        if mk0.p < 0.01:
            label1 = 'slope = %.3f p < 0.01'%(mk0.slope)
        else:
            label1 = 'slope = %.3f p = %.2f'%(mk0.slope,mk0.p)
        ax.text(x, y, label1,
                transform = ax.transAxes, color = c, size = 8)
    
    if label == 'L_n':
        ax.set_ylim(-.3, 3.6)
        ax.legend((l1,l2,l3), ['OBS','ANT + NAT', 'NAT'], 
                  loc='lower right', prop={'size':7}, ncol = 3)

axes[0,0].set_title('CDHW event in low-income countries')
axes[0,1].set_title('CDHW event in high-income countries')

# plot detection and attribution
def func(x, low = True):
    unc = np.percentile(x, [25, 75])
    unc = unc[1] - unc[0]
    q1 = np.percentile(x, 25) - unc * 1.5
    q3 = np.percentile(x, 75) + unc * 1.5
    x1 = x[(x>=q1)&(x<=q3)]
    unc = (np.min(x1), np.max(x1))
    
    q1 = unc[0]
    q3 = unc[1]
    if low:
        return q1
    else:
        return q3

beta.variable = pd.Categorical(beta.variable, categories=['L','LM','UM','H'],
                               ordered=True)
x = ['L','LM','UM','H']
y1 = beta.loc[beta.type=='ANT',:].groupby('variable').mean().value.values
y2 = beta.loc[beta.type=='NAT',:].groupby('variable').mean().value.values
ylow1 = beta.loc[beta.type=='ANT',:].groupby('variable').agg(func).value.values
yupp1 = beta.loc[beta.type=='ANT',:].groupby('variable').agg(lambda x: func(x, False)).value.values
ylow2 = beta.loc[beta.type=='NAT',:].groupby('variable').agg(func).value.values
yupp2 = beta.loc[beta.type=='NAT',:].groupby('variable').agg(lambda x: func(x, False)).value.values
yerr1 = np.abs(np.vstack([ylow1, yupp1]) - y1)
yerr2 = np.abs(np.vstack([ylow2, yupp2]) - y2)

ax1 = axes[1,0]
trans1 = Affine2D().translate(-0.1, 0.0) + ax1.transData
trans2 = Affine2D().translate(+0.1, 0.0) + ax1.transData
er1 = ax1.errorbar(x, y1, yerr=yerr1, marker="o", linestyle="none", 
                   transform=trans1, color = '#C44B31', capsize = 4)
er2 = ax1.errorbar(x, y2, yerr=yerr2, marker="o", linestyle="none", 
                   transform=trans2, color = '#2F84F2', capsize = 4)
ax1.set_ylabel('Scaling factors')
ax1.legend().set_title(None)
ax1.set_xlabel(None)
ax1.axhline(y = 0, ls = (5, (12,)), c = 'k', lw = .3)
ax1.axhline(y = 1, ls = (5, (12,)), c = 'k', lw = .3)
ax1.xaxis.set_minor_locator(ticker.NullLocator())
ax1.yaxis.set_major_locator(ticker.FixedLocator(np.arange(-.2, 1.2, .2)))
ax1.legend((er1, er2), ('ANT','NAT'), loc = 'lower right')
ax1.set_xticklabels(['LIC','LMIC','UMIC','HIC'])

obsT = {}
for label in ['L','LM','UM','H']:
    obs0 = dfObs[label+'DH'].values
    mk0 = mk.original_test(obs0)
    obsT[label] = [mk0.slope,]
obsT = pd.DataFrame.from_dict(obsT).T.reset_index()
obsT.columns = ['variable','value']
obsT['type'] = 'OBS'
attr = pd.concat([attr[['variable','type','value']], obsT])

attr = attr.loc[attr.type!='OBS',:]
ax2 = axes[1,1]
sns.barplot(data=attr, x = 'variable', y = 'value', hue = 'type',
            ax = ax2, errorbar = ('pi', 90), width = .6,
            palette = ['#C44B31','#2F84F2', '#CECCCC'])
ax2.set_ylabel('Attributable trends')
ax2.legend().set_title(None)
ax2.set_xlabel(None)
ax2.axhline(y = 0, ls = 'dashed', c = 'k', lw = .3)
ax2.xaxis.set_minor_locator(ticker.NullLocator())
ax2.set_xticklabels(['LIC','LMIC','UMIC','HIC'])

fig.tight_layout()
for name,ax in zip(['a','b','c','d'], axes.ravel()):
    ax.text(-.15, 1, name, weight='bold', transform = ax.transAxes)
    
fig.savefig('c:/Research/fig3.svg')

