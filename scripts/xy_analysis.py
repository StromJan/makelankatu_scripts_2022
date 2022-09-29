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
    
    cmap = cm.romaO_r
    cmap2 = cm.vikO
    
    
    
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





plot_temps(False)



