#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:09:44 2021

@author: stromjan
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import osmnx as ox
import pyproj
import os
import warnings
from cmcrameri import cm
import netCDF4 as nc


good_colors = ['#ac0535','#e85e45','#fbbf6c','#fcfdb9','#bde4a2','#54afac','#654a9c','#851170'][::-1]#,'#8d377d'][::-1]


def make_colormap(N,colors=good_colors,linear=True,bad='white'):
    if (linear):
        colmap = mpl.colors.LinearSegmentedColormap.from_list('name',colors,N)
        colmap.set_bad('lightgrey')
    else:
        colmap = mpl.colors.ListedColormap(colors)
        colmap.set_bad('lightgrey')
    return colmap


warnings.filterwarnings("ignore")

os.chdir("/home/stromjan/Output/")

palm_path = '/media/stromjan/Work_Drive/Puhti/JOBS/0609_morning_{}'
fixed = '_fixed_flow'

cmc = make_colormap(8)

file1 = np.load('/home/stromjan/Scripts/First_paper/Validation/gridded_sniffer_201706090716.npz',allow_pickle=True)
file2 = np.load('/home/stromjan/Scripts/First_paper/Validation/gridded_sniffer_201706090823.npz',allow_pickle=True)

cols = file1['cols']
lons = file1['longitude']
lats = file1['latitude']
# Grid spacing
dx = file1['dx']

print (dx)

print (cols)

# Coordinates as northing and easting:
Ns = np.arange( 0, dx*len(lats), dx ) + 6675862.079
Es = np.arange( 0, dx*len(lons), dx ) + 25497100.010

temps1 = file1['mean'][:,:,cols=='CPC_tot']*1E6
temps2 = file2['mean'][:,:,cols=='CPC_tot']*1E6


temps = np.nanmean(np.dstack((temps1,temps2)),axis=-1)


runs = ['off_noprocesses'+fixed, 'off_processes'+fixed, 'on_noprocesses'+fixed, 'on_processes'+fixed]
runsi = ['S0', 'S1', 'R0', 'R1']
runsi = ['\mathrm{R_0A_0}','\mathrm{R_0A_1}', '\mathrm{R_1A_0}', '\mathrm{R_1A_1}']




timep = np.arange(24)
ntot_mean   = np.zeros( [len( runs ), len( lats ), len( lons )] )
ntot_median = np.zeros( [len( runs ), len( lats ), len( lons )] )
ntot_p90    = np.zeros( [len( runs ), len( lats ), len( lons )] )


print('      interpolate to the Sniffer grid')

for r in range(4):
    
    ds_palm = nc.Dataset( '{}/OUTPUT/0609_morning_{}_av_masked_N03_M04.MEAN2.nc'.format(palm_path.format(runs[r]),runs[r]))

    zi = 1 # 0 = 1 m, 1 = 2 m, 2 = 4 m height




    # timep = ds_palm.variables['time'][:]
    xp = ds_palm.variables['x'][:]
    yp = ds_palm.variables['y'][:]
    zp = ds_palm.variables['ku_above_surf'][:]

    Np = ds_palm.variables['N_UTM'][:]
    Ep = ds_palm.variables['E_UTM'][:]
    
    palm_proj = pyproj.Proj( 'epsg:3879' )
    lons_palm, lats_palm = palm_proj( Ep, Np, inverse=True )

    salsa_Ntot = np.nanmean(ds_palm.variables['salsa_Ntot'][:2,:,:], axis=0)

    salsa_Ntot[salsa_Ntot<0] = np.nan

    ds_palm.close()
# Interpolate PALM data to the Sniffer grid
    for i in range( len( lons ) ):
      ind1 = ( (Ep>Es[i]) & ( Ep<=Es[i]+dx ) )
      temp = np.copy(salsa_Ntot[:,ind1])

      for j in range( len( lats ) ):
        ind2 = ( (Np>Ns[j]) & ( Np<=Ns[j]+dx ) )
        ntot_mean[r,j,i] = np.nanmean(np.nanmean(temp[ind2,:], axis=-1), axis=-1)
        ntot_median[r,j,i] = np.nanmedian(np.nanmean(temp[ind2,:], axis=-1), axis=-1)
        ntot_p90[r,j,i] = np.nanpercentile(np.nanmean(temp[ind2,:], axis=-1), 90, axis=-1)



ntot_mean[:,np.isnan(temps)] = np.nan


ntot_median[:,np.isnan(temps)] = np.nan




[print (np.nanmean((v-temps)/temps*100)) for v in ntot_mean[:]]
[print (np.nanmean((np.abs(v-temps))/np.abs(temps)*100)) for v in ntot_mean[:]]






fig, ax = plt.subplots(figsize=(16,5),nrows=1,ncols=len(runs)+1,constrained_layout=True,sharex='col',sharey='row')

cmap1 = plt.cm.rainbow
cmap1 = cm.hawaii_r
cmap2 = plt.cm.RdBu_r

cmap1.set_bad('none')
cmap2.set_bad('none')

LON, LAT = np.meshgrid( lons, lats )
place_name = 'Vallila, Helsinki, Finland'

print('Load in Open Street Map data')







graph = ox.graph_from_place(place_name,network_type='drive')
nodes, edges = ox.graph_to_gdfs(graph)

buildings = ox.geometries_from_place(place_name,tags={'building':True})


print('... done!')

    
ax[0].set_title(r'Obs. ($cm^{-3}$)',fontsize=18)




im = ax[0].pcolormesh(LON, LAT, temps*1E-6, cmap=cmap1, alpha=0.9, zorder=100,norm=mpl.colors.LogNorm(vmin=1E3, vmax=5E5))
ax[0].set_title('Obs.',fontsize=18)


for ir in range(len(runs)):

    im2 = ax[ir+1].pcolormesh(LON, LAT, (ntot_mean[ir]-temps)/temps*100, cmap=cmap2, alpha=0.9, zorder=100, vmin=-150, vmax=150)
    
    
    print (np.nanmax((ntot_mean[ir]-temps)/temps*100))
    ax[ir+1].set_title(r'${}$'.format(runsi[ir]),fontsize=18)



txts = ['a)','b)','c)','d)','e)']

for i,axi in enumerate(ax):
    axi.set_facecolor('gray')

    buildings.plot(ax=axi, facecolor='black')
    axi.set_xlim( 24.947, 24.956 )
    axi.set_xticks( np.arange( 24.948, 24.955, 0.004 ) )
    axi.set_ylim( 60.195, 60.199 )
    axi.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
    axi.set_yticks( [ 60.196, 60.198 ] )
    axi.plot(24.952011, 60.196433,'k*',markersize=14,linewidth=1, zorder=100, markerfacecolor=cmc(0.5), alpha=1, label='SR1')
    axi.tick_params(axis='both', labelsize=16)
    axi.text(0.88, 0.9, txts[i],transform=axi.transAxes,
            fontsize=18,bbox=dict(facecolor='0.8', edgecolor='none', pad=3.0))

cbar1 = fig.colorbar(im,ax=ax[0],ticks=[1E2,1E3,1E4,1E5],aspect=10,orientation='horizontal')
cbar2 = fig.colorbar(im2,ax=ax[1:],aspect=40,orientation='horizontal')
cbar1.ax.tick_params(labelsize=16) 
cbar2.ax.tick_params(labelsize=16) 
cbar1.set_label(r'$\mathrm{N_{tot,2m} (cm^{-3})}$',rotation=0,fontsize=18)
cbar2.set_label(r'$\Delta \mathrm{N_{tot,2m}} (\%)$',rotation=0,fontsize=18)

fig.savefig('/home/stromjan/Output/Obs_and_bias_Ntot_revision2.png',dpi=300)
fig.savefig('/home/stromjan/Output/Obs_and_bias_Ntot_revision2.pdf',dpi=300)


