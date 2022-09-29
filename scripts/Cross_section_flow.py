#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 17:54:07 2021
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
from scipy.spatial.transform import Rotation
from scipy.stats.mstats import gmean
from scipy import ndimage, misc,stats
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import gaussian_filter
import pandas as pd
import time
import cmocean
from mapTools import *
from cmcrameri import cm
import seaborn as sns
sns.set()
pal = sns.hls_palette(h=.9)
plt.style.use('default')
import os
os.chdir("/home/stromjan/Output/")

def read_var(nc,var):
    return nc.variables[var][:]


def coordrot(u,v,theta):
    A = np.pi/180*theta
    ur = u*np.cos(A) + v*np.sin(A)
    vr = -u*np.sin(A) + v*np.cos(A)
    return ur,vr


var = 'u'
var2 = 'v'
var3 = 'w'
var4 = 'theta'


cmap = SCM6.romaO_r



anim = False



domain = 'N03'

period = ['first','last']

p = 0

fixed = '_fixed_flow'

directory = '/media/stromjan/Work_Drive/Puhti/JOBS/{}/OUTPUT/{}'

mask1 = '_av_masked_{}_M01.MEAN.nc'
mask4 = '_av_masked_{}_M04.MEAN.nc'


R1_name = '0609_morning_on_processes'+fixed
R0_name = '0609_morning_on_noprocesses'+fixed


S1_name = '0609_morning_off_processes'+fixed
S0_name = '0609_morning_off_noprocesses'+fixed


R0  =  NetCDFFile(directory.format(R0_name,R0_name   + mask1.format(domain)))
R0_aero  =  NetCDFFile(directory.format(R0_name,R0_name   + mask4.format(domain)))

R1  =  NetCDFFile(directory.format(R1_name,R1_name   + mask1.format(domain)))
R1_aero  =  NetCDFFile(directory.format(R1_name,R1_name   + mask4.format(domain)))




S0  =  NetCDFFile(directory.format(S0_name,S0_name   + mask1.format(domain)))
S0_aero  =  NetCDFFile(directory.format(S0_name,S0_name   + mask4.format(domain)))


S1  =  NetCDFFile(directory.format(S1_name,S1_name   + mask1.format(domain)))
S1_aero  =  NetCDFFile(directory.format(S1_name,S1_name   + mask4.format(domain)))



height = 20

u_vals = np.asarray([S0.variables['u'][:height],R1.variables['u'][:height],R0.variables['u'][:height],R1.variables['u'][:height]])
v_vals = np.asarray([S0.variables['v'][:height],R1.variables['v'][:height],R0.variables['v'][:height],R1.variables['v'][:height]])
w_vals = np.asarray([S0.variables['w'][:height],R1.variables['w'][:height],R0.variables['w'][:height],R1.variables['w'][:height]])

var = 'salsa_Ntot'

Ntot_vals = np.asarray([S0_aero.variables[var][:height],S1_aero.variables[var][:height],R0_aero.variables[var][:height],R1_aero.variables[var][:height]])




masks = np.copy(Ntot_vals)

u_vals[u_vals<-100] = -101
v_vals[v_vals<-100] = -101
w_vals[w_vals<-100] = -101



x1, x2 = (18,32)
y1 = 90
z1, z2 = (0,26)
current_cmap = cm.hawaii_r
current_cmap.set_bad(color='black')

current_cmap2 = plt.cm.RdBu_r
current_cmap2.set_bad(color='black')

titles = ['S0','S1','R0','R1']
titles = ['\mathrm{R_0A_0}','\mathrm{R_1A_1}', '\mathrm{R_1A_0}', '\mathrm{R_1A_1}','\mathrm{R_1A_1}*']





buildings = (masks<=0)

masks*= 0
masks[buildings] = np.nan

masks2 = np.copy(w_vals)


u_vals[np.abs(u_vals)>10] = 0
v_vals[np.abs(v_vals)>10] = 0
w_vals[np.abs(w_vals)>10] = 0

buildings2 = (masks2<-100)

masks2*= 0
masks2[buildings2] = 100

theta = 51
for i in range(4):


    Ntot_vals[i] = ndimage.rotate(np.copy(Ntot_vals[i]), theta, reshape=False, axes=(1,2))
    masks[i] = ndimage.rotate(np.copy(masks[i]), theta, reshape=False, axes=(1,2))


    u_vals[i] = ndimage.rotate(np.copy(u_vals[i]), theta, reshape=False, axes=(1,2))
    v_vals[i] = ndimage.rotate(np.copy(v_vals[i]), theta, reshape=False, axes=(1,2))
    w_vals[i] = ndimage.rotate(np.copy(w_vals[i]), theta, reshape=False, axes=(1,2))
    masks2[i] = ndimage.rotate(np.copy(masks2[i]), theta, reshape=False, axes=(1,2))



u_vals[np.abs(u_vals)>10] = np.nan
v_vals[np.abs(v_vals)>10] = np.nan
w_vals[np.abs(w_vals)>10] = np.nan



idx2 = masks2>1


idx3 = Ntot_vals[0]<0







idx = binary_dilation(masks>1,iterations=1)
idx = gaussian_filter(idx,sigma=0.2)
masks[idx] = 1
masks[~idx] = 0


txts = ['a)', 'b)', 'c)', 'd)']
tcolors = ['black','black','black','black']
    
plot = 'rotated'

fig, axes = plt.subplots(ncols=2,sharey=True,figsize=(14,8),constrained_layout=True)

if (plot=='cross'):
    
    x1, x2 = (44,64)
    x,y = np.meshgrid(np.arange(170),np.arange(150))
    
    horiz = 1
    vert = 1
    
    mask = (((x*vert-horiz*y)==47)) #& (x<60) & (x>40))
    
    
    for i in range(2):
    
    

        x,z = np.meshgrid(np.arange(x1,x2),np.arange(z1,z2))
    
    
        u_masked = u[:,mask]
        v_masked = v[:,mask]
        w_masked = w_vals[i][:,mask]


        im = axes[i].imshow(w_masked[z1:z2,x1:x2],vmin=-1,vmax=1,origin='lower',cmap=current_cmap)
        c = axes[i].streamplot(x-x1,z-z1,u_masked[z1:z2,x1:x2],w_masked[z1:z2,x1:x2],density=0.6,color='k')
        c.lines.set_alpha(0.5)
        for ar in axes[i].get_children():
            if type(ar)==mpl.patches.FancyArrowPatch:
                ar.set_alpha(0.5) # or x.set_visible(False)
        axes[i].set_xlabel('x(m)',fontsize=12)
        axes[i].set_title(titles[i],fontsize=12)

elif (plot=='rotated'):

    x,y = np.meshgrid(np.arange(170),np.arange(150))

    vert = []
    
    axx = axes.ravel().tolist()


    for i,ax in enumerate(axes.flatten()):
        
        
            u,v = coordrot(u_vals[i],v_vals[i],theta)
            w = w_vals[i]
            Ntot = Ntot_vals[i]

            
            w[:8][idx3[:8]] = np.nan
            
            print (Ntot.shape,w.shape)


            
            c0,c00 = (120,297) #40,128
            r0,r00 = (146,193) #69,88

            c1,c2 = (40,128) #40,128
            r1,r2 = (69,88) #69,88
            


            off = 0
            step = 2
            u = np.mean(u[off:,c0:c00,r0:r00:step],axis=1)
            v = np.mean(v[off:,c0:c00,r0:r00:step],axis=1)
            


            w = np.mean(w[off:,c0:c00,r0:r00:step],axis=1)
            Ntot = np.mean(Ntot[off:,c0:c00,r0:r00:step],axis=1)

            print (Ntot.shape,w.shape)
            x,z = np.meshgrid(np.arange(len(Ntot[0])),np.arange(len(Ntot)))

            print (titles[i],np.nanmean(Ntot))
            print ('mean: {}, max: {}, min: {}'.format(np.nanmean(Ntot),np.nanmax(Ntot),np.nanmin(Ntot)))

            vert.append(np.nanmean(Ntot))

            c = ax.streamplot(x,z,u,w,density=0.6,color='k')
            c.lines.set_alpha(0.5)
            for ar in ax.get_children():
                if type(ar)==mpl.patches.FancyArrowPatch:
                    ar.set_alpha(0.5) # or x.set_visible(False)
            


            
            im = ax.imshow(w,origin='lower',cmap=current_cmap2,vmin=-0.7,vmax=0.7)
            base = w
            
            print (np.nanmean(w[:,:13]),np.nanmean(w[:,12:]))
            
            



            labels = np.arange(len(Ntot[0]))
            labels2 = np.arange(len(Ntot))
            ax.text(0.88, 0.85, txts[i], color=tcolors[i],transform=ax.transAxes,
                fontsize=18,bbox=dict(facecolor='0.8', edgecolor='none', pad=3.0))
            

            
            
            
            

            ax.set_xlabel('x(m)',fontsize=18)



            ax.set_title(r'${}$'.format(titles[i]),fontsize=18)


            
            
            ax.set_xticks(labels[::2])
            ax.set_xticklabels(['{:.0f}'.format(2*s) for s in labels[::2]])
                
            ax.set_yticks(labels2[::2])
            ax.set_yticklabels(['{:.0f}'.format(2*s) for s in labels2[::2]])
            
            ax.tick_params(axis='both', which='major', labelsize=16)

            ax.set_aspect('equal')

    

    

else:

    

    for i in range(2):
        im = axes[i].imshow(w_vals[i][z1:z2,y1,x1:x2],vmin=-1,vmax=1,cmap=current_cmap,origin='lower')

        axes[i].set_xlabel('x(m)',fontsize=12)
        axes[i].set_title(titles[i],fontsize=12)
    
    
axes.ravel()[0].set_ylabel('z(m)',fontsize=18)

 



cbar = fig.colorbar(im, ax=axes.ravel().tolist(),orientation='horizontal',aspect=40)
cbar.set_label('w (m/s)',rotation=0,fontsize=20)
cbar.ax.tick_params(labelsize=18) 

fig.savefig('Rotated_Wind_NEW.pdf',dpi=250)
fig.savefig('Rotated_wind_NEW.png',dpi=250)



