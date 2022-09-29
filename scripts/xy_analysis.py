#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:14:45 2020

@author: stromjan
"""


from netCDF4 import Dataset as NetCDFFile
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.dates as md
import matplotlib as mpl
import datetime as dt
from datetime import timedelta
import dateutil.relativedelta as relativedelta
from matplotlib import colors
from scipy.signal import detrend
import pandas as pd
import time
import cmocean
from mapTools import *

from cmcrameri import cm
import seaborn as sns
from useful_funcs import *
import os
os.chdir("/home/stromjan/Output/")

sns.set()
pal = sns.hls_palette(h=.9)
plt.style.use('default')


def read_var(nc,var):
    return nc.variables[var][:]





directory = '/media/stromjan/Work_Drive/Puhti/JOBS/{}/OUTPUT/{}'

file = '_av_xy_{}.COMBINED.nc'

fixed = '_fixed_flow'


R1_name = '0609_morning_on_processes' + fixed
R0_name = '0609_morning_on_noprocesses' + fixed



S1_name = '0609_morning_off_processes'+fixed
S2_name = '0609_morning_off_noprocesses'+fixed



var = 'salsa_PM2.5'


domain = 'N03'

cmap = mpl.cm.Spectral_r






R1 =  NetCDFFile(directory.format(R1_name,R1_name + file.format(domain)))
R0 =  NetCDFFile(directory.format(R0_name,R0_name + file.format(domain)))


region1 = [R1, R0]



var = 'u_xy'
vaR0 = 'v_xy'
vaRD0 = 'w_xy'
var4 = 'theta_xy'
var5 = 'tsurf*_xy'
var6 = 'us*_xy'


cmap = cm.vikO_r


base = R0
comp = R1

[print (v) for v in R1.VAR_LIST.split(';')]





def plot_temps(save=False):
    tsurf_rad_4734 = rad_4734.variables['tsurf*_xy'][:]-273.15
    
    
    
    
    origin = pd.to_datetime(rad_4734.origin_time) + dt.timedelta(hours=3)
    
    
    times = [origin + dt.timedelta(seconds=t) for t in rad_4734.variables['time'][:]]
    times = pd.to_datetime(times)
    timestrs = times[:].strftime('%H:%M')
    
    cmap = SCM6.romaO_r
    cmap2 = SCM6.vikO
    
    
    
    fig, ax = plt.subplots(ncols=3,sharey=True,figsize=(14,6),constrained_layout=True)
    
    
    inds = [0,-1]
    
    for i,ind in enumerate(inds):
        
        surf = ax[i].imshow(tsurf_rad_4734[ind,0,:,:],origin='lower',cmap=cmap,vmin=10,vmax=36)
        ax[i].set_title('{}'.format(timestrs[ind]),fontsize=14)
        ax[i].set_xlabel('x(m)',fontsize=12)
    
    ax[0].set_ylabel('y(m)',fontsize=12)    
    
    
    cbar = fig.colorbar(surf, ax=ax[:-1].ravel().tolist(),orientation='horizontal',aspect=30)
    cbar.set_label(r'$\degree C$',fontsize=14)
    
    

    rad_tdiff = (tsurf_rad_4734[-1,0,:,:]-tsurf_rad_4734[0,0,:,:])
    
        
    

    im2 = ax[-1].imshow(rad_tdiff,origin='lower',cmap=cmap2,norm=mpl.colors.TwoSlopeNorm(vcenter=0,vmin=-15,vmax=15))#,vmin=10,vmax=35)

    ax[-1].set_xlabel('x(m)',fontsize=12)

    
    ax[-1].set_title(r'$\Delta T$ ({} - {})'.format(timestrs[-1],timestrs[0]),fontsize=14)
    cbaR0 = fig.colorbar(im2,ax=[ax[-1]],orientation='horizontal',aspect=15)
    cbaR0.set_label(r'$\Delta\degree C$',fontsize=14)
    


    if (save):
        fig.savefig('Temperatures_all.png',dpi=250)



def plot_net_rad(save=False):

    tsurf_rad_4734 = rad_4734.variables['rad_sw_in*_xy'][:]
    
    

    
    origin = pd.to_datetime(rad_4734.origin_time) + dt.timedelta(hours=3)
    
    
    times = [origin + dt.timedelta(seconds=t) for t in rad_4734.variables['time'][:]]
    times = pd.to_datetime(times)
    timestrs = times[:].strftime('%H:%M')
    
    cmap = SCM6.romaO_r
    
    
    
    fig, ax = plt.subplots(ncols=2,sharey=True,figsize=(12,10),constrained_layout=True)
    
    
    
    for i in range(2):
        
        surf = ax[i].imshow(tsurf_rad_4734[-i,0,:,:],origin='lower',cmap=plt.cm.RdBu_r,vmin=0,vmax=530)
        ax[i].set_title('Surface temperatures at {}'.format(timestrs[-i]),fontsize=14)
        ax[i].set_xlabel('x(m)',fontsize=12)
    
    ax[0].set_ylabel('y(m)',fontsize=12)    
    
    
    cbar = fig.colorbar(surf, ax=ax.ravel().tolist(),orientation='horizontal',aspect=30)
    cbar.set_label(r'$\degree C$',fontsize=14)
    
    
    fig2, ax2 = plt.subplots(figsize=(12,10),constrained_layout=True)

    rad_tdiff = (tsurf_rad_4734[-1,0,:,:]-tsurf_rad_4734[0,0,:,:])
    
        
    
    im = ax2.imshow(rad_tdiff,origin='lower',cmap=cmap)#,norm=mpl.colors.TwoSlopeNorm(vcenter=0))#,vmin=10,vmax=35)
    
    
    ax2.set_title('Surface temperature change between {}-{}'.format(timestrs[0],timestrs[-1]),fontsize=14)
    cbaR0 = fig2.colorbar(im,ax=[ax2],orientation='horizontal',shrink=0.70)
    cbaR0.set_label(r'$\Delta\degree C$',fontsize=14)
    
    ax2.set_xlabel('x(m)',fontsize=12)
    ax2.set_ylabel('y(m)',fontsize=12)

    if (save):
        fig.savefig('Temperatures.png',dpi=250)
        fig2.savefig('Temperatures_diff.png',dpi=250)



def plot_rad_change(var):    
    h = 0
    ts = 12
    te = -1
    
    titles = ['R0', 'R1']
    titles2 = ['S0', 'S1']
    cs = 60
    ce = 521
    
    base = R0.variables[var][:][ts,h,cs:ce,cs:ce]

    mean1 = np.mean(R0.variables[var][:][ts:te,h,cs:ce,cs:ce],axis=0)
    mean2 = np.mean(R1.variables[var][:][ts:te,h,cs:ce,cs:ce],axis=0)
    mean3 = np.mean(RD0.variables[var][:][ts:te,h,cs:ce,cs:ce],axis=0)
    
    fig, axes = plt.subplots(figsize=(14,14),ncols=2,nrows=2,constrained_layout=True)
    
    ax = axes.flatten()
    d0 = R0.variables[var][:][te,h,cs:ce,cs:ce] - base
    d1 = R1.variables[var][:][te,h,cs:ce,cs:ce] - base
    dd = RD0.variables[var][:][te,h,cs:ce,cs:ce] - base

    deltas = [d0,d1,dd]

    

    
    cmap = cm.vik
    
    cmap.set_bad('grey')
    
    pot = 0
    
    

    im1 = ax[0].imshow(base-273.15,origin='lower',cmap=cm.roma_r,vmin=10,vmax=30)
    ax[0].set_title('Initial T_surf (R0)',fontsize=14)
    
    for i,d in enumerate(deltas):
        im2 = ax[i+1].imshow(d,origin='lower',cmap=cmap,vmin=-5,vmax=15)

    
    
    for i,a in enumerate(ax[1:]):
        a.set_title(f'R0_ts - {titles[i]}_tf',fontsize=14)
    
    cbar1 = fig.colorbar(im1, ax=ax[:2],orientation='horizontal',aspect=30)
    cbar2 = fig.colorbar(im2, ax=ax[1:],orientation='horizontal',aspect=30,shrink=0.95)


    cbar1.set_label(r'$^o C$',horizontalalignment='right',labelpad=20,rotation=0,fontsize=18)
    cbar2.set_label(r'$\Delta ^o C$',horizontalalignment='right',labelpad=20,rotation=0,fontsize=18)


    plt.savefig('tsurf_diff.png')

def plot_sw_change(var):    
    h = 0
    ts = -27
    te = -1
    
    titles = ['R0', 'R1', 'RD0']
    titles2 = ['S0', 'S1']
    cs = 0 #60
    ce = -1 #521
    

    mean1 = np.nanmean(R0.variables[var][:][ts:te,h,cs:ce,cs:ce],axis=0)
    mean2 = np.nanmean(R1.variables[var][:][ts:te,h,cs:ce,cs:ce],axis=0)





    
    print (np.nanmax(mean2),np.nanmean(mean2))
    
    fig, axes = plt.subplots(figsize=(18,8),ncols=3,constrained_layout=True)
    
    ax = axes.flatten()

    d1 = R1.variables[var][:][0,h,cs:ce,cs:ce]
    dd = R1.variables[var][:][-1,h,cs:ce,cs:ce]

    deltas = [d1,dd]

    

    
    cmap = cm.vik
    
    cmap.set_bad('grey')
    
    pot = 0
    
    

    im1 = ax[0].imshow(mean1,origin='lower',cmap=cm.roma_r,vmin=0,vmax=np.nanmax(mean2))
    ax[0].set_title(r'R0 mean SW$_{in}$',fontsize=14)
    
    
    
    
    
    for i,d in enumerate(deltas):
        im2 = ax[i+1].imshow(d,origin='lower',cmap=cmap,vmin=0,vmax=np.nanmax(deltas))
        print (np.nanmean(d))

    
    times = ['07:00','09:15']
    
    for i,a in enumerate(ax[1:]):
        a.set_title('Net radiation at {}'.format(times[i]),fontsize=14)
    
    cbar1 = fig.colorbar(im1, ax=ax[:1],orientation='horizontal',aspect=15)
    cbar2 = fig.colorbar(im2, ax=ax[1:],orientation='horizontal',aspect=30,shrink=1.0)


    cbar1.set_label(r'$W/m^{2}$',horizontalalignment='right',labelpad=20,rotation=0,fontsize=18)


    plt.savefig('sw_diff.png')


plot_temps(False)



