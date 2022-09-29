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
import ScientificColourMaps6 as SCM6
import seaborn as sns
from useful_funcs import *
from scipy.spatial.transform import Rotation
from scipy import ndimage, misc,stats
from cmcrameri import cm
import sys
import os
import pyproj
os.chdir("/home/stromjan/Output/")


sns.set()
pal = sns.hls_palette(h=.9)
plt.style.use('default')


def read_var(nc,var):
    return nc.variables[var][:]


rev = ''




directory = '/media/stromjan/Work_Drive/Puhti/JOBS/{}/OUTPUT/{}'



mask1 = '_av_masked_{}_M01.MEAN.nc'
mask4 = '_av_masked_{}_M04.MEAN.nc'
fixed = '_fixed_flow'



R1_name = '0609_morning_on_processes'+fixed
R0_name = '0609_morning_on_noprocesses'+fixed






print ('base: ',R0_name,' comp: ',R1_name)


S1_name = '0609_morning_off_processes'+fixed
S0_name = '0609_morning_off_noprocesses'+fixed

version = ''

local = True



var = 'u'
var2 = 'v'
var3 = 'w'
var4 = 'theta'

domain = 'N03'

cmap = mpl.cm.Spectral_r



anim = False


R11 =  NetCDFFile(directory.format(R1_name,R1_name + mask1.format(domain)))
R01 =  NetCDFFile(directory.format(R0_name,R0_name + mask1.format(domain)))



S11 =  NetCDFFile(directory.format(S1_name,S1_name + mask1.format(domain)))
S01 =  NetCDFFile(directory.format(S0_name,S0_name + mask1.format(domain)))







times = read_var(R11,'time')
start = -27 #int(7200/300)
end = None



j = 0

winds = [R11, R01, S11, S01]

titles = ['R1','R0','S1','S0']
titles2 = ['\mathrm{R_1A_1}','\mathrm{R_1A_0}', '\mathrm{R_0A_1}', '\mathrm{R_0A_0}']

base_i = -1
comp_i = 0


base1 = winds[base_i]


comp1 = winds[comp_i]



xp = base1.variables['x'][:]
yp = base1.variables['y'][:]
zp = base1.variables['ku_above_surf'][:]

Np = base1.variables['N_UTM'][:]
Ep = base1.variables['E_UTM'][:]

palm_proj = pyproj.Proj( 'epsg:3879' )
lons_palm, lats_palm = palm_proj( Ep, Np, inverse=True )

LON, LAT = np.meshgrid( lons_palm, lats_palm )



u_arrs = [read_var(base1,var)[:,::-1,:],read_var(comp1,var)[:,::-1,:]]
v_arrs = [read_var(base1,var2)[:,::-1,:],read_var(comp1,var2)[:,::-1,:]]
w_arrs = [read_var(base1,var3)[:,::-1,:],read_var(comp1,var3)[:,::-1,:]]
T_arrs = [read_var(base1,var4)[:,::-1,:],read_var(comp1,var4)[:,::-1,:]]






heights = '{} m above surface'




#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
def plot_wind(vertical=True,diff=False,other=None):

    
    txts = ['a)', 'b)', 'c)', 'd)']
    tcolors = ['black','black','black','black']
    n=14
    nc = 2
    nr = 2
    
    

    X, Y = np.meshgrid(np.arange(0,461,n),np.arange(0,461,n))


    j = 1

    
    
    
    

    if (not diff):
        fig, axes = plt.subplots(ncols=3,sharex=True,sharey=True,figsize=(14,6),constrained_layout=True)
        
        stat = []
        
        for i in range(2):

            u = u_arrs[i][j,::-1]
            v = v_arrs[i][j,::-1]
            w = w_arrs[i][j,::-1]
            t = T_arrs[i][j,::-1]
            


            stat.append(np.hypot(u,v))
            q = axes[i].quiver(LON[::n,::n],LAT[::n,::n],u[::n,::n],v[::n,::n],scale=50)

            
            if (vertical):
                cmap = plt.cm.bwr
                cmap.set_bad('dimgrey')
    
                surf = axes[i].imshow(w,cmap=cmap,vmin=-1,vmax=1)

            elif (other is not None):
                if (other=='tke'):
                    cmap = plt.cm.Spectral_r
                    cmap.set_bad('dimgrey')

                    
                    surf = axes[i].imshow(tke,cmap=cmap,vmin=0,vmax=0.7)
                    

                elif (other=='us'):
                    cmap = plt.cm.Spectral_r
                    cmap.set_bad('dimgrey')

                    surf = axes[i].imshow(us,cmap=cmap,vmin=0,vmax=0.5)


                elif (other=='div'):
                    cmap = plt.cm.RdBu_r
                    cmap.set_bad('dimgrey')

                    surf = axes[i].imshow(div,cmap=cmap,vmin=-0.15,vmax=0.15)
                    
                elif (other=='momflux'):
                    cmap = plt.cm.Spectral_r
                    cmap.set_bad('dimgrey')
                    uw[uw==0] = np.nan

                    surf = axes[i].imshow(uw,cmap=cmap,vmin=0,vmax=0.25)


                elif (other=='heat'):
                    cmap = plt.cm.Spectral_r
                    cmap.set_bad('dimgrey')

                    surf = axes[i].imshow(wt,cmap=cmap,vmin=0,vmax=0.2)

                elif (other=='theta'):
                    cmap = cm.roma_r
                    cmap.set_bad('dimgrey')

                    surf = axes[i].imshow(t-273.15,cmap=cmap,vmin=12,vmax=16)


                elif (other=='T_change'):
                    cmap = plt.cm.RdBu_r
                    cmap.set_bad('dimgrey')
                    t = T_arrs[i][-1,j]-T_arrs[i][0,j]
                    levs = np.arange(-4.5,4.5,0.25)

                    surf = axes[i].contourf(t,cmap=cmap,levels=levs,origin='upper')
                    q = axes[i].quiver(X,Y,u[::-n,::n],v[::-n,::n],scale=50)
                    axes[i].set_ylim(0,len(u))
                    axes[i].set_xlim(0,len(u[0]))

                elif (other=='test'):
                    cmap = plt.cm.Spectral_r
                    cmap.set_bad('dimgrey')

                    t = T_arrs[i][-1,j]
                    levs = np.arange(280,292,0.5)
                    

                    surf = axes[i].contourf(t,cmap=cmap,levels=levs,origin='upper')
                    q = axes[i].quiver(X,Y,u[::-n,::n],v[::-n,::n],scale=50)
                    t[~np.isnan(t)] = np.nan
                    axes[i].imshow(t,cmap=cmap)
                    axes[i].set_ylim(0,len(u))
                    axes[i].set_xlim(0,len(u[0]))


                    
            else:
                cmap = plt.cm.Spectral_r
                cmap.set_bad('dimgrey')
                surf = axes[i].imshow(np.hypot(u,v),origin='lower',cmap=cmap,vmin=0,vmax=2,extent=[LON[0,0], LON[0,-1], LAT[0,0],LAT[-1,0]])
                        


        



        axes[0].set_ylabel('Latitude ($^\circ$)',fontsize=12)  

        [axes[i].set_title(r'${}$'.format(titles2[j]),fontsize=14) for i,j in enumerate([base_i,comp_i])]


        cbar = fig.colorbar(surf, ax=axes[:2].ravel().tolist(),orientation='horizontal',aspect=30)
        cbar.set_label(r'$\mathrm{V_{4m} (m~s^{-1})}$',fontsize=14)

        base = stat[0]
        comp = stat[1]
        

        

        delta = (((comp-base)/base)*100)
        delta[np.abs(delta)<0.1] = np.nan
        

        lower = [delta.min(), -100, delta.max()]
        higher = [delta.min(), 100, delta.max()]
  

    
        cmap2 = SCM6.vikO
        cmap2.set_bad('dimgrey')
        axes[-1].set_facecolor('dimgrey')


        bounds = np.arange(-0.6, 1.31, 0.1)


        print ('base mean: {}, comp mean: {}'.format(np.nanmean(base),np.nanmean(comp)))
        print ('base max: {}, comp max: {}'.format(np.nanmax(base),np.nanmax(comp)))
        print ('base min: {}, comp min: {}'.format(np.nanmin(base),np.nanmin(comp)))

        print ('percentage change comp-base {}'.format((np.nanmean(comp)-np.nanmean(base))/np.nanmean(base)*100))

        p = axes[-1].contourf(LON,LAT,comp-base, levels=bounds, cmap=cmap2, norm=mpl.colors.TwoSlopeNorm(vcenter=0,vmin=-0.6, vmax=1.3))

    

        axes[-1].set_title(r'$\Delta({} - {})$'.format(titles2[comp_i],titles2[base_i]),fontsize=14)


        cbar2 = fig.colorbar(p,ax=[axes[-1]],orientation='horizontal',aspect=15)
        cbar2.set_label(r'$\mathrm{\Delta(m~s^{-1})}$',fontsize=14)
        
        cbar.ax.tick_params(labelsize=12) 
        cbar2.ax.tick_params(labelsize=12) 
        
        for i in range(3):
            axes[i].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
            axes[i].set_yticks( [ 60.196, 60.197, 60.198, 60.199])

            axes[i].text(0.88, 0.9, txts[i], color=tcolors[i],transform=axes[i].transAxes,
                    fontsize=18,bbox=dict(facecolor='0.8', edgecolor='none', pad=3.0))
            
            axes[i].tick_params(axis='both', labelsize=12)
            axes[i].set_xlabel('Longitude ($^\circ$)',fontsize=12)  
            
        axes[0].set_aspect(2)
        axes[1].set_aspect(2)
        axes[-1].set_aspect(2)
        

        plt.savefig('Wind_difference_{}_vs_{}_4m_NEW.pdf'.format(titles[base_i],titles[comp_i]),dpi=250)
        plt.savefig('Wind_difference_{}_vs_{}_4m_NEW.png'.format(titles[base_i],titles[comp_i]),dpi=250)

            
    else:
        fig, axes = plt.subplots(figsize=(10,10),constrained_layout=True)

        h = 1
        



        uv4734_rad = np.sqrt(u_arrs[0][h]**2+v_arrs[0][h]**2)
        uv4734_chem = np.sqrt(u_arrs[1][h]**2+v_arrs[1][h]**2)


        w4734_rad = w_arrs[0][h]
        w4734_chem = w_arrs[1][h]


        surf = axes.imshow(w4734_rad-w4734_chem,cmap=SCM6.vikO,norm=colors.TwoSlopeNorm(vcenter=0))
        cbar = fig.colorbar(surf,orientation='vertical')
        axes.set_title(f'Difference in vertical wind ({2**h}m) (RRTMG On - Off)',fontsize=12)




def plot_temp(T_arrs):


    n=14
    nc = 3
 
    X, Y = np.meshgrid(np.arange(0,461,n),np.arange(0,461,n))
    fig, axes = plt.subplots(ncols=nc,figsize=(14,6),constrained_layout=True)


    run = 0
    j = 0
    
    cmap = SCM6.romaO_r
    cmap.set_bad('dimgrey')

    timestrs = ['07:05','09:15']

    for i in range(2):


        axes[i].set_title('{}'.format(timestrs[i]),fontsize=14)
        axes[i].set_xlabel('x(m)',fontsize=12)
    
    
    T1 = T_arrs[run][0,j]-273.15#-T_arrs[i][0,1]
    T2 = T_arrs[run][-1,j]-273.15#-T_arrs[i][0,1]
        
    surf = axes[0].imshow(T1,cmap=cmap,vmin=9,vmax=16)
    surf = axes[1].imshow(T2,cmap=cmap,vmin=9,vmax=16)



    axes[0].set_ylabel('y(m)',fontsize=12)    
    
    
    cbar = fig.colorbar(surf, ax=axes[:-1].ravel().tolist(),orientation='horizontal',aspect=30)
    cbar.set_label(r'$\degree C$',fontsize=14)
    
    

    dt = (T2-T1)
    
        
    
    im2 = axes[-1].imshow(dt,cmap=SCM6.vikO,vmin=-4,vmax=4)#,norm=mpl.colors.TwoSlopeNorm(vcenter=0))#,vmin=10,vmax=35)
    axes[-1].set_xlabel('x(m)',fontsize=12)  
    axes[-1].set_title(r'$\Delta T$ ({} - {})'.format(timestrs[-1],timestrs[0]),fontsize=14)
    cbar2 = fig.colorbar(im2,ax=[axes[-1]],orientation='horizontal',aspect=15)
    cbar2.set_label(r'$\Delta\degree C$',fontsize=14)
    cbar.ax.tick_params(labelsize=12) 
    cbar2.ax.tick_params(labelsize=12) 


def plot_all_temps():

    
    txts = ['a)', 'b)', 'c)', 'd)']
    tcolors = ['black','black','black','black']


    n = 12


    X, Y = np.meshgrid(np.arange(0,461,n),np.arange(0,461,n))
    

    fbf = False
    j = 0
    
    if (fbf):
        fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(15,12),constrained_layout=True)
        
        
        axes = ax.flatten()
        stat = []
        base = read_var(winds[3],var4)[0,::-1,:]
        
        
        
        
        
        
        
        for i,ax in enumerate(axes.flatten()):
            u = np.mean(read_var(winds[::-1][i],var)[0,::-1,:], axis=0)
            v = np.mean(read_var(winds[::-1][i],var2)[0,::-1,:], axis=0)
            w = np.mean(read_var(winds[::-1][i],var3)[0,::-1,:], axis=0)
            T = (np.mean(read_var(winds[::-1][i],var4)[0,::-1,:], axis=0))
            
            
            
            
            
            q = ax.quiver(X,Y,u[::n,::n],v[::n,::n],scale=50)
            
            cmap = SCM6.lajolla
            cmap.set_bad('dimgrey')
            
            
            cmap2 = cm.roma_r
            cmap2.set_bad('dimgrey')
            
            if (i==0):
            
                plot1 = ax.imshow(T-273.15,cmap=cmap,vmin=8.4,vmax=8.8)#norm=colors.LogNorm(vmin=0.01,vmax=5))
                cbar1 = fig.colorbar(plot1, ax=axes[0],orientation='vertical',aspect=30)
                fig.text(0.48,0.75,r'$T_{2m}^{o}C$',fontsize=14,rotation=0)
                
            
            else:
                plot2 = ax.imshow(T-base,cmap=cmap2,vmin=0,vmax=5)#norm=colors.LogNorm(vmin=0.01,vmax=5))
    
    
            ax.set_title(titles2[i],fontsize=16)
            ax.tick_params(axis='both', labelsize=14)
            ax.text(0.88, 0.9, txts[i], color=tcolors[i],transform=ax.transAxes,
                    fontsize=18,bbox=dict(facecolor='0.8', edgecolor='none', pad=3.0))

    
    
            
        cbar2 = fig.colorbar(plot2,ax=axes[1:],orientation='vertical',aspect=30)
        cbar2.ax.set_title(r'$\Delta^{o}C$',fontsize=16)

        cbar2.ax.tick_params(labelsize=14)
    else:
        
        fig, axes = plt.subplots(ncols=3,sharey=True,sharex=True,figsize=(14,6),constrained_layout=True)
        
        stat = []
        current_cmap = cm.roma_r
        current_cmap.set_bad('dimgrey')

        for i in range(2):
            
            u = u_arrs[i][j,::-1]
            v = v_arrs[i][j,::-1]

            T = T_arrs[i][j,::-1] -273.15
            stat.append(T)
            q = axes[i].quiver(LON[::n,::n],LAT[::n,::n],u[::n,::n],v[::n,::n],scale=50)
            


            
            surf = axes[i].imshow(T,origin='lower',vmin=8,vmax=13,cmap=current_cmap,extent=[LON[0,0], LON[0,-1], LAT[0,0],LAT[-1,0]])
            print ('Mean T2m: ',np.nanmean(T))
            print ('Max T2m: ',np.nanmax(T))
            print ('Min T2m: ',np.nanmin(T))

            
        
        
        axes[0].set_ylabel('Latitude ($^\circ$)',fontsize=12)  

        [axes[i].set_title(r'${}$'.format(titles2[j]),fontsize=14) for i,j in enumerate([base_i,comp_i])]



        cbar = fig.colorbar(surf, ax=axes[:2].ravel().tolist(),orientation='horizontal',aspect=30)
        cbar.set_label(r'$T_{2m} (^oC)$',fontsize=14)
        


    
        cmap2 = SCM6.lajolla
        cmap2.set_bad('dimgrey')
        axes[-1].set_facecolor('dimgrey')


        base = stat[0]
        comp = stat[1]

        bounds = np.arange(2.5, 5.01, 0.25)

        print ('Smallest change in T2m: ',np.nanmin(comp-base))
        p = axes[-1].contourf(LON,LAT,comp-base, levels=bounds, cmap=cmap2, vmin=2.5, vmax=5)
        print ('Mean change in T2m: ',np.nanmean(comp-base))


        axes[-1].set_title(r'$\Delta({} - {})$'.format(titles2[comp_i],titles2[base_i]),fontsize=14)



        cbar2 = fig.colorbar(p,ax=[axes[-1]],ticks=np.arange(2.5,5.01,0.25)[::2],orientation='horizontal',aspect=15)
        cbar2.set_label(r'$\Delta(^oC)$',fontsize=14)
        
        

        for i in range(3):

            axes[i].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
            axes[i].set_yticks( [ 60.196, 60.197, 60.198, 60.199])





            axes[i].text(0.88, 0.9, txts[i], color=tcolors[i],transform=axes[i].transAxes,
                    fontsize=18,bbox=dict(facecolor='0.8', edgecolor='none', pad=3.0))
            
            axes[i].tick_params(axis='both', labelsize=12)

            axes[i].set_xlabel('Longitude ($^\circ$)',fontsize=12)  


    cbar.ax.tick_params(labelsize=12) 
    cbar2.ax.tick_params(labelsize=12) 
    axes[0].set_aspect(2)
    axes[1].set_aspect(2)
    


    plt.savefig('theta_2m_comparison_NEW.png',dpi=250)
    plt.savefig('theta_2m_comparison_NEW.pdf',dpi=250)

    return



plot_wind(False,False)


plot_all_temps()
