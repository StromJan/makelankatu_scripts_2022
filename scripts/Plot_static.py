#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 09:30:48 2022

@author: stromjan
"""
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as NetCDFFile
import matplotlib as mpl
from matplotlib.colors import LightSource
from cmcrameri import cm
import os
import pyproj
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

os.chdir("/home/stromjan/Output/")
good_colors = ['#ac0535','#e85e45','#fbbf6c','#fcfdb9','#bde4a2','#54afac','#654a9c','#851170'][::-1]


def make_colormap(N,colors=good_colors,linear=True,bad='white'):
    if (linear):
        colmap = matplotlib.colors.LinearSegmentedColormap.from_list('name',colors,N)
        colmap.set_bad('lightgrey')
    else:
        colmap = matplotlib.colors.ListedColormap(colors)
        colmap.set_bad('lightgrey')
    return colmap


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def create_ticks(values,labels):
    cticks = []
    
    for v in np.unique(values):
        cticks.append(labels[v])
    
    return cticks



vegetation_labels = ['None','Bare soil (no vegetation)', 'Crops, mixed farming', 'Short grass', 'Evergreen needleleaf trees', 
                     'Deciduous needleleaf trees', 'Evergreen broadleaf trees', 'Deciduous broadleaf trees', 'Tall grass', 'Desert', 
                     'Tundra', 'Irrigated crops', 'Semidesert', 'Ice caps and glaciers', 'Bogs and marshes', 'Evergreen shrubs', 
                     'Deciduous shrubs', 'Mixed forest / woodland', 'Interrupted forest']

pavement_labels = ['None','Unknown pavement', 'Asphalt', 'Concrete', 
                   'Sett', 'Paving stoned', 'cobblestone', 'Metal', 'Wood', 'Gravel', 'Fine gravel', 'Pebblestone', 'Woodchips', 
                   'Tartan', 'Artificial turf', 'Clay (sports)', 'Building (dummy)']

water_labels = ['None','Lake', 'River', 'Ocean', 'Pond', 'Fountain']


building_labels = ['None','Residential (-1950)', 'Residential (1951-2000)', 'Residential (2001-)', 'Office (-1950)', 
                   'Office (1951-2000)', 'Office (2001-)']




soil_labels = ['Coarse', 'Medium', 'Medium-fine', 'Fine', 'Very fine', 'Organic']

fname = '/media/stromjan/Work_Drive/Run_Preparation/PIDS_STATIC_N03'

static = NetCDFFile(fname)

print (static)
num = 4
cmap = make_colormap(num)


pavement    = static['pavement_type'][:]
water       = static['water_type'][:]
vegetation  = static['vegetation_type'][:]
buildings   = static['building_type'][:]
soil        = static['soil_type'][:]
lad         = static['lad'][:]
zt          = static['zt'][:]
buildings_2d   = static['buildings_2d'][:]



plots  = [static['building_type'][:],static['pavement_type'][:],static['vegetation_type'][:],static['soil_type'][:]]
labels = [building_labels, pavement_labels, vegetation_labels,soil_labels]
plot_labels = ['Building type', 'Pavement type', 'Vegetation type','Soil type']









bld_labels = create_ticks(np.unique(plots[0])[:-1], labels[0])
pav_labels = create_ticks(np.unique(plots[1])[:-1], labels[1])
veg_labels = create_ticks(np.unique(plots[2])[:-1], labels[2])
soi_labels = create_ticks(np.unique(plots[3])[:-1], labels[3])


labels = [bld_labels, pav_labels, veg_labels, soi_labels]

for i,p in enumerate(plots):
    uniq = np.unique(p)[:-1]

    for j,u in enumerate(uniq):
        p[p==u] = j
        








print (bld_labels,pav_labels,veg_labels)




directory   = '/media/stromjan/Work_Drive/Puhti/{}/OUTPUT/{}'
file        = '_av_xy_{}.COMBINED.nc'
data_name   = '0609_morning_off_processes'
data        =  NetCDFFile(directory.format(data_name,data_name + file.format('N03')))



Np                   = data.variables['N_UTM'][:]
Ep                   = data.variables['E_UTM'][:]
palm_proj            = pyproj.Proj( 'epsg:3879' )
lons_palm, lats_palm = palm_proj( Ep, Np, inverse=True )
LON, LAT             = np.meshgrid( lons_palm, lats_palm )




for i,plot in enumerate(plots):
    
    fig, ax = plt.subplots(figsize=(14,12),constrained_layout=True)
    
    
    
    uniq = np.unique(plot)[:-1]
    
    uniq = uniq-1

    
    cmap = plt.cm.Spectral_r  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, len(uniq))
    
    cmap.set_bad('gray')
    
    im = plt.imshow(plot,origin='lower',cmap=cmap,extent=[LON[0,0], LON[0,-1], LAT[0,0],LAT[-1,0]],interpolation='bilinear')
    
    
    cbar=fig.colorbar(im,ax=[ax],ticks=range(len(uniq)),location='right',shrink=.89,aspect=30,pad=0.05)
    
    print (uniq)
    

    plt.clim(-0.5, len(uniq) - 0.5)

    ticks = create_ticks(range(len(uniq)), labels[i])

    cbar.ax.set_yticklabels(ticks,fontsize=18)
    cbar.ax.tick_params(labelsize=16) 

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xticks(np.arange(24.948, 24.955, 0.004))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
    ax.set_yticks( [ 60.196, 60.198 ] )
    ax.set_ylabel(r'Latitude ($\circ$)',fontsize=20)
    ax.set_xlabel(r'Longitude ($\circ$)',fontsize=20)

    ax.set_title(plot_labels[i],fontsize=22)
    ax.set_aspect('auto')

### added
#100m at latitude 60.2 in x-direction is 6371000m*cos(60.2 deg)*sin(0.001809595025 deg)

    scalebar = AnchoredSizeBar(ax.transData,
                           0.001809595025, '100 m', 'lower left', 
                           pad=1,
                           color='black',
                           frameon=True,
                           size_vertical=0.00001,alpha=0.2)

    ax.add_artist(scalebar)
###

    plt.savefig('{}_revision2.png'.format(plot_labels[i].split(' ')[0]),dpi=250)
    plt.savefig('{}_revision2.pdf'.format(plot_labels[i].split(' ')[0]),dpi=250)



def terrainplot():
    

    
    
    
    
    fig, ax = plt.subplots(figsize=(10,10),constrained_layout=True)
    im1 = ax.imshow(zt,origin='lower',cmap='viridis',vmin=0)
    im2 = ax.imshow(buildings_2d,origin='lower',cmap='binary',vmin=0)
    ############### ADDED
#    ls = LightSource(azdeg=99.91,altdeg=80.21)
    # shade data, creating an rgb array.
#    rgb = ls.shade(buildings_2d,plt.cm.Reds)
#    im3 = ax.imshow(rgb,origin='lower')
    #####################
    cbar1 = fig.colorbar(im1,ax=[ax],location='right',aspect=20,shrink=0.77,pad=.03)
    cbar2 = fig.colorbar(im2,ax=[ax],location='left',aspect=20,shrink=0.77,pad=.03)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    
    
    ax.set_xticks(np.arange(24.948, 24.955, 0.004))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
    ax.set_yticks( [ 60.196, 60.198 ] )

    
    
    cbar1.set_label('Orography (m)',fontsize=14)
    cbar2.set_label('Topography (m)',fontsize=14)

    plt.savefig('Terrain.png',dpi=250)
    

    plt.show()
    

def plot():
    mask = np.copy(water)*0

    
    
    mask[buildings_2d > 0] = 0
    mask[pavement > 0] = 1
    mask[vegetation > 0] = 2
    mask[lad[4] > 0] = 3
    
    

    ticks = ['Buildings','Pavement','Low vegetation','Trees']#,'Water']
    

    colors = ['#010203', '#2C3640','#016064','#008000']
    colors = ['#010203','#7E7E7E','#A0E989','#008000']
              
              
    cmap3 = make_colormap(num,colors=colors)
    
    
    azimuth7 = 74.1
    azimuth9 = 103.41

    orig1, orig2 = (269,227)
    x1,y1 = (269,227)
    x2,y2 = (269,227)
    sunrise_az = 35.78
    sunset_az = 324.25
    x3,y3 = (405,300)
    x4,y4 = (238,287)
    
    length = 2.6
    
    x1 += length*90*np.cos((90-azimuth7)*np.pi/180)
    y1 += length*90*np.sin((90-azimuth7)*np.pi/180)
    x2 += length*90*np.cos((azimuth9-90)*np.pi/180) 
    y2 += -length*90*np.sin((azimuth9-90)*np.pi/180)
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    im = plt.imshow(mask,origin='lower',cmap=cmap3,alpha=.85)
    cbar=fig.colorbar(im,ax=[ax],ticks=range(num),location='top',shrink=.89,aspect=30,pad=0.05)
    plt.clim(-0.5, num - 0.5)
    
    cmc = make_colormap(8)
    
    
    SR1 = plt.plot(313,172,'k*',markersize=14,linewidth=1, markerfacecolor=cmc(0.5), alpha=1, label='SR1')
    SR2 = plt.plot(332,200,'k*',markersize=14,linewidth=1, markerfacecolor='cyan',alpha=1, label='SR2')
    SR3 = plt.plot(516,283,'k*',markersize=14,linewidth=1, markerfacecolor='magenta', alpha=1, label='SR3')

   # facecolor='#ac0535',alpha=1, label='SR3')
    
    cbar.ax.set_xticklabels(ticks,fontsize=14)
    rect = mpl.patches.Rectangle((324,155), 46, 180, linewidth=1, 
                                  edgecolor='black', facecolor='magenta',alpha=0.5, angle=51,label='Averaged cross-section',fill='none',hatch='+')
    ax.add_patch(rect)

    style = "Fancy, tail_width=6, head_width=14, head_length=14"
    kw1 = dict(arrowstyle=style, color="yellow")
    kw2 = dict(arrowstyle=style, color="black")
    ax.plot([orig1,x1],[orig2,y1],lw=3,color='orange',zorder=1)
    ax.plot([orig1,x2],[orig2,y2],lw=3,color='orange',zorder=1)
    
    
    sun1 = mpl.patches.Circle((x1,y1), 10, linewidth=2, 
    
                                  edgecolor='orange', facecolor='yellow',alpha=1)
    sun2 = mpl.patches.Circle((x2,y2), 10, linewidth=2, 
                                  edgecolor='orange', facecolor='yellow',alpha=1)
    
    run = mpl.patches.FancyArrowPatch((x1+15,y1+15), (x2+15,y2-15),
                                 connectionstyle="arc3,rad={}".format(-(azimuth9-azimuth7)*np.pi/180), **kw1,label='Sun\'s path (07:00 - 09:15 UTC+3)')

    
    ax.add_patch(run)

    
    mpl
    patches = [sun1,sun2]
    coll = mpl.collections.PatchCollection(patches,edgecolor='orange',facecolor='yellow',linewidth=2,zorder=10)
    ax.add_collection(coll)

    
    ax.set_xlabel('Longitude ($^\circ$)',fontsize=16)
    ax.set_ylabel('Latitude ($^\circ$)',fontsize=16)
    

    fig.legend(bbox_to_anchor=(0.0,.88,1.,0.),loc='upper center', ncol=5,fontsize=12,facecolor='grey',framealpha=0.3,columnspacing=1)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    
    # Axis:
    ax.set_rasterized(True)
    ax.set_yticklabels(['{:.3f}'.format(x) for x in LAT[::576//7,0]])
    ax.set_xticklabels(['{:.3f}'.format(x) for x in LON[0,::576//7]])
    
    


    ax.yaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
    ax.yaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter('%.3f'))    
    
### added
    scalebar = AnchoredSizeBar(ax.transData,
                           100, '100 m', 'lower left', 
                           pad=1,
                           color='black',
                           frameon=True,
                           size_vertical=1,alpha=0.2)

    ax.add_artist(scalebar)
###

    plt.savefig('Regions_and_vortex_area_revision2.png',dpi=250,bbox_inches='tight')
    plt.savefig('Regions_and_vortex_area_revision2.pdf',bbox_inches='tight')

    plt.show()

plt.close('all')
plot()
# terrainplot()

