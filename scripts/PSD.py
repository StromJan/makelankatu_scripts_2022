#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 11:13:00 2021

@author: stromjan
"""


from netCDF4 import Dataset as NetCDFFile
import numpy as np
import xarray as xr
import pyproj
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
import ScientificColourMaps6 as SCM6
import seaborn as sns
from useful_funcs import *
from scipy.spatial.transform import Rotation
from scipy import ndimage, misc,stats
import sys
import psdLib
from psdLib import define_bins
import os




def find_nearest(array, value):
    """ Find the nearest value in an array and return also the index """
  
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    return array[idx], idx


sns.set()
pal = sns.hls_palette(h=.9)
plt.style.use('default')

fixed = '_fixed_flow'



directory = '/media/stromjan/Work_Drive/Puhti/JOBS/{}/OUTPUT/{}'

mask1 = '_av_masked_{}_M01.MEAN.nc'
mask4 = '_av_masked_{}_M04.MEAN.nc'

R0_name = '0609_morning_on_noprocesses' +fixed
R1_name = '0609_morning_on_processes'+fixed



S0_name = '0609_morning_off_noprocesses'+fixed
S1_name = '0609_morning_off_processes'+fixed

version = ''

local = True



var = 'salsa_PM2.5'


domain = 'N03'

cmap = mpl.cm.Spectral_r




R0 =  NetCDFFile(directory.format(R0_name,R0_name + mask4.format(domain)))
R1 =  NetCDFFile(directory.format(R1_name,R1_name + mask4.format(domain)))



S0 =  NetCDFFile(directory.format(S0_name,S0_name + mask4.format(domain)))
S1 =  NetCDFFile(directory.format(S1_name,S1_name + mask4.format(domain)))




aero  = [R0, R1, S0, S1]

titles = ['R0','R1', 'S0','S1']
titles = [r'$\mathrm{R_1A_0}$',r'$\mathrm{R_1A_1}$', r'$\mathrm{R_0A_0}$',r'$\mathrm{R_0A_1}$']


base_i = -2
comp_i = 0

ts = -27
te = None





x = np.arange(461)
y = np.arange(461)
nbins = 10
zi = 1 #1 = 4m, 7= 16m

reglim = [2.50E-9, 1.50E-8, 1.0E-6]
nbin = [3,7]
dmid, bin_limits = define_bins( nbin, reglim)


coord_bg = [24.+57.3386/60., 60.+11.8432/60]
coord_os = [24.+57.1602/60., 60.+11.7925/60]
coord_ss = [24.+57.1025/60., 60.+11.7947/60]





bg = np.zeros( [len( titles ), nbins] )
ss = np.zeros( [len( titles ), nbins] )
os = np.zeros( [len( titles ), nbins] )
cn = np.zeros( [len( titles ), nbins] )


for fi, ds_palm in enumerate(aero):
  

  # Load in PALM data

  
    timep = ds_palm.variables['time'][:]
    xp = ds_palm.variables['x'][:]
    yp = ds_palm.variables['y'][:]
    zp = ds_palm.variables['ku_above_surf'][:]
  
    Np = ds_palm.variables['N_UTM'][:]
    Ep = ds_palm.variables['E_UTM'][:]
    
    for ib in range( nbins ):
        if ib==0:
            psd = np.zeros([ len(yp), len(xp), nbins ])
        psd[:,:,ib] = ds_palm.variables['salsa_N_bin{}'.format(ib+1)][zi,:,:]
    
    ds_palm.close()  
  
  # Change northing and easting in PALM data to latitude and longitude
    if fi==0:
        temp_psd_coarse = np.zeros( [len( y ), len( x ), nbins] )
        palm_proj = pyproj.Proj( 'epsg:3879' )
        lons_palm, lats_palm = palm_proj( Ep, Np, inverse=True )
    elif fi>1:
        dummy_lon, dummy_lat = palm_proj( Ep, Np, inverse=True )
        if np.logical_or( dummy_lat != lats_palm, dummy_lon != lons_palm ).any():
            print('Latitudes and longitudes of the PALM output files do not match!')
            sys.exit(1)






    bgy = find_nearest(lats_palm, coord_bg[1])[1]
    bgx = find_nearest(lons_palm, coord_bg[0])[1]

    ssy = find_nearest(lats_palm, coord_ss[1])[1]
    ssx = find_nearest(lons_palm, coord_ss[0])[1]
    
    osy = find_nearest(lats_palm, coord_os[1])[1]
    osx = find_nearest(lons_palm, coord_os[0])[1]


    bgi = np.nanmean(psd[bgy-2:bgy+3,bgx-2:bgx+3,:],axis=(0,1))
    ssi = np.nanmean(psd[ssy-2:ssy+3,ssx-2:ssx+3,:],axis=(0,1))
    osi = np.nanmean(psd[osy-2:osy+3,osx-2:osx+3,:],axis=(0,1))


    buildings = (psd<=0)
    
    psd_rot = ndimage.rotate(np.copy(psd), 51, reshape=False, axes=(0,1))
    mask_rot = ndimage.rotate(np.copy(buildings), 51, reshape=False, axes=(0,1))
    psd_rot[mask_rot] = np.nan

    
    cni = np.nanmean(psd_rot[120:300,169:192,:],axis=(0,1)) #192
    psd_rot,mask_rot = (None,None)

    


  # Select the matching times and data points with the measurement data
    
    
    temp = temp_psd_coarse[:,:,:]

    bg[fi,:] = bgi
    ss[fi,:] = ssi
    os[fi,:] = osi
    cn[fi,:] = cni



fig, axess = plt.subplots(figsize=(14,16),nrows=2, ncols=2,constrained_layout=True)
lines = ['--rs','-rs', '--bs','-bs']
dummylines = ['--r','-r', '--b','-b']

dummylines2 = ['rs','rs', 'bs','bs']

txts = ['a)', 'b)', 'c)', 'd)']


axes = axess.flatten()

for i in range(len(lines)):
    ssp = ss[i]*1e-6 / np.log10( bin_limits[1::] / bin_limits[0:-1] )
    osp = os[i]*1e-6 / np.log10( bin_limits[1::] / bin_limits[0:-1] )
    bgp = bg[i]*1e-6 / np.log10( bin_limits[1::] / bin_limits[0:-1] )
    cnp = cn[i]*1e-6 / np.log10( bin_limits[1::] / bin_limits[0:-1] )


    axes[0].loglog(dmid,ssp,dummylines[i],label=r'{}'.format(titles[i]))
    axes[1].loglog(dmid,osp,dummylines[i])
    axes[2].loglog(dmid,bgp,dummylines[i])
    axes[3].loglog(dmid,cnp,dummylines[i])
    
    axes[0].loglog(dmid,ssp,dummylines2[i])#,label=r'${}$'.format(titles[i]))
    axes[1].loglog(dmid,osp,dummylines2[i])
    axes[2].loglog(dmid,bgp,dummylines2[i])
    axes[3].loglog(dmid,cnp,dummylines2[i])
    
    print (cnp)


    


    
axes[0].legend(loc='best',prop={'size': 16})

for i,ax in enumerate(axes):
    ax.text(0.02, 0.95, txts[i], transform=ax.transAxes, fontsize=18)
    ax.text(0.02, 0.95, txts[i], transform=ax.transAxes,
                        fontsize=18,bbox=dict(facecolor='0.8', edgecolor='none', pad=3.0))
    ax.grid()
    ax.set_xlabel('$D_{\mathrm{mid}}$ (m)',fontsize=18)
    ax.set_ylim(0.5e1, 5e5 )
    ax.set_ylabel( r'$dN/d\logD~\mathrm{(cm^{-3})}$',fontsize=18)
    ax.get_xaxis().set_tick_params(which='major', labelsize=18)
    ax.get_yaxis().set_tick_params(which='major', labelsize=18)



plt.savefig('/home/stromjan/Output/PSD_{}m_Canyon_4x4_NEW.png'.format(2+ zi*2),dpi=250)
plt.savefig('/home/stromjan/Output/PSD_{}m_Canyon_4x4_NEW.pdf'.format(2+ zi*2),dpi=250)


