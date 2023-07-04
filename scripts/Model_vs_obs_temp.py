#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:20:10 2021

@author: stromjan
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import osmnx as ox
import pyproj
import warnings
from cmcrameri import cm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import netCDF4 as nc

warnings.filterwarnings("ignore")


good_colors = ['#ac0535','#e85e45','#fbbf6c','#fcfdb9','#bde4a2','#54afac','#654a9c','#851170'][::-1]#,'#8d377d'][::-1]


def make_colormap(N,colors=good_colors,linear=True,bad='white'):
    if (linear):
        colmap = mpl.colors.LinearSegmentedColormap.from_list('name',colors,N)
        colmap.set_bad('lightgrey')
    else:
        colmap = mpl.colors.ListedColormap(colors)
        colmap.set_bad('lightgrey')
    return colmap



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

# Coordinates as northing and easting:
Ns = np.arange( 0, dx*len(lats), dx ) + 6675862.079
Es = np.arange( 0, dx*len(lons), dx ) + 25497100.010

temps1 = file1['mean'][:,:,cols=='T']
temps2 = file2['mean'][:,:,cols=='T']


temps = np.nanmean(np.dstack((temps1,temps2)),axis=-1)


runs = ['off_noprocesses'+fixed, 'on_processes'+fixed, 'on_noprocesses'+fixed, 'on_processes'+fixed]
runsi = ['S0', 'R1', 'R0', 'R1']
runsi = ['\mathrm{R_0A_0}','\mathrm{R_1A_1}', '\mathrm{R_1A_0}', '\mathrm{R_1A_1}']




timep = np.arange(24)
ntot_mean   = np.zeros( [len( runs ), len( lats ), len( lons )] )
ntot_median = np.zeros( [len( runs ), len( lats ), len( lons )] )
ntot_p90    = np.zeros( [len( runs ), len( lats ), len( lons )] )

print('      interpolate to the Sniffer grid')

for r in range(2):
    
    ds_palm_temp = nc.Dataset( '{}/OUTPUT/0609_morning_{}_av_masked_N03_M01.MEAN2.nc'.format(palm_path.format(runs[r]),runs[r]))

    zi = 1# 0 = 0.5 m, 1 = 1.5 m, 2 = 2.5 m, 3 = 3.5 m height




    # timep = ds_palm.variables['time'][:]
    xp = ds_palm_temp.variables['x'][:]
    yp = ds_palm_temp.variables['y'][:]
    zp = ds_palm_temp.variables['ku_above_surf'][:]

    Np = ds_palm_temp.variables['N_UTM'][:]
    Ep = ds_palm_temp.variables['E_UTM'][:]
    
    palm_proj = pyproj.Proj( 'epsg:3879' )
    lons_palm, lats_palm = palm_proj( Ep, Np, inverse=True )

    salsa_temp = np.nanmean(ds_palm_temp.variables['theta'][:2,:,:],axis=0)-273.15 # black carbon

    salsa_temp[salsa_temp<0] = np.nan


    ds_palm_temp.close()
# Interpolate PALM data to the Sniffer grid
    for i in range( len( lons ) ):
      ind1 = ( (Ep>Es[i]) & ( Ep<=Es[i]+dx ) )
      temp = np.copy(salsa_temp[:,ind1])

      for j in range( len( lats ) ):
        ind2 = ( (Np>Ns[j]) & ( Np<=Ns[j]+dx ) )
        ntot_mean[r,j,i] = np.nanmean(np.nanmean(temp[ind2,:], axis=-1), axis=-1)
        ntot_median[r,j,i] = np.nanmedian(np.nanmean(temp[ind2,:], axis=-1), axis=-1)




ntot_mean[:,np.isnan(temps)] = np.nan


[print (np.nanmean(v-temps)) for v in ntot_mean[:]]


fig, ax = plt.subplots(figsize=(12,5),nrows=1,ncols=3,constrained_layout=True,sharex='col',sharey='row')

cmap1 = cm.lajolla
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

    
ax[0].set_title('Obs.',fontsize=16)


im = ax[0].pcolormesh(LON, LAT, temps, cmap=cmap1, alpha=0.9, zorder=100,vmin=12,vmax=13)

print('Observations mean, max, min: {}, {}, {}'.format(np.nanmean(temps),np.nanmax(temps),np.nanmin(temps)))

for ir in range (2):
    plot = ntot_mean[ir]-temps
    idx = plot<-100
    


    im2 = ax[ir+1].pcolormesh(LON, LAT, plot, cmap=cmap2, alpha=0.9, zorder=1000, vmin=-4, vmax=4)
    print('mean, max, min: {}, {}, {}'.format(np.nanmean(ntot_mean[ir]),np.nanmax(ntot_mean[ir]),np.nanmin(ntot_mean[ir])))

    
    ax[ir+1].set_title('${}$ - Obs.'.format(runsi[ir]),fontsize=16)


txts = ['a)','b)','c)','d)','e)']

for i,axi in enumerate(ax):
    axi.set_facecolor('gray')

    buildings.plot(ax=axi, facecolor='black')
    axi.set_xlim( 24.947, 24.956 )
    axi.set_xticks( np.arange( 24.948, 24.955, 0.004 ))
    axi.set_ylim( 60.195, 60.199 )
    axi.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
    axi.set_yticks( [ 60.196, 60.198 ])
    axi.plot(24.952011, 60.196433,'k*',markersize=14,linewidth=1, zorder=1000, markerfacecolor=cmc(0.5), alpha=1, label='SR1')
    axi.tick_params(axis='both', labelsize=14)
    axi.text(0.88, 0.9, txts[i],transform=axi.transAxes,
            fontsize=18,bbox=dict(facecolor='0.8', edgecolor='none', pad=3.0))


### added
#100m at latitude 60.2 in x-direction is 6371000m*cos(60.2 deg)*sin(0.001809595025 deg)

    scalebar = AnchoredSizeBar(ax[0].transData,
                           0.001809595025, '100 m', 'lower left', 
                           pad=0.5,
                           color='black',
                           frameon=True,
                           size_vertical=0.00001,alpha=0.2)
    
    ax[0].add_artist(scalebar)
###
       

cbar1 = fig.colorbar(im,ax=ax[0],aspect=20,orientation='horizontal')
cbar2 = fig.colorbar(im2,ax=ax[1:],aspect=40,orientation='horizontal')
cbar1.ax.tick_params(labelsize=14) 
cbar2.ax.tick_params(labelsize=14) 
cbar1.set_label(r'$\mathrm{T}_{2m} (^\circ C)$',rotation=0,fontsize=16)
cbar2.set_label(r'$\Delta \mathrm{T_{2m}}(^\circ C)$',rotation=0,fontsize=16)
plt.savefig('/home/stromjan/Output/Obs_and_bias_Temp_revision2.pdf',dpi=300)
plt.savefig('/home/stromjan/Output/Obs_and_bias_Temp_revision2.png',dpi=300)
