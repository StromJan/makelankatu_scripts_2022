#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 18:16:56 2021

@author: stromjan
"""

from netCDF4 import Dataset as NetCDFFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import ndimage, misc,stats
import os
from cmcrameri import cm
import pyproj

os.chdir("/home/stromjan/Output/")



good_colors = ['#ac0535','#e85e45','#fbbf6c','#fcfdb9','#bde4a2','#54afac','#654a9c','#851170'][::-1]#,'#8d377d'][::-1]


def make_colormap(N,colors=good_colors,linear=True,bad='white'):
    if (linear):
        colmap = mpl.colors.LinearSegmentedColormap.from_list('name',colors,N)
        colmap.set_bad('lightgrey')
    else:
        colmap = mpl.colors.ListedColormap(colors)
        colmap.set_bad('lightgrey')
    return colmap


sns.set()
pal = sns.hls_palette(h=.9)
plt.style.use('default')

def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def read_var(nc,var):
    global units
    out = nc.variables[var]
    units = out.units
#    print (units)
    return out[:]


rev = ''

fixed = '_fixed_flow'



directory = '/media/stromjan/Work_Drive/Puhti/JOBS/{}/OUTPUT/{}'

mask1 = '_av_masked_{}_M01.MEAN.nc'
mask4 = '_av_masked_{}_M04.MEAN.nc'


R1_name = '0609_morning_on_processes'+fixed
R0_name = '0609_morning_on_noprocesses'+fixed


S1_name = '0609_morning_off_processes'+fixed
S0_name = '0609_morning_off_noprocesses'+fixed

version = ''

local = True



var = 'salsa_PM2.5'

domain = 'N03'


cmap = mpl.cm.Spectral_r



R11 =  NetCDFFile(directory.format(R1_name,R1_name + mask1.format(domain)))
R01 =  NetCDFFile(directory.format(R0_name,R0_name + mask1.format(domain)))



S11 =  NetCDFFile(directory.format(S1_name,S1_name + mask1.format(domain)))
S01 =  NetCDFFile(directory.format(S0_name,S0_name + mask1.format(domain)))

R14 =  NetCDFFile(directory.format(R1_name,R1_name + mask4.format(domain)))
R04 =  NetCDFFile(directory.format(R0_name,R0_name + mask4.format(domain)))



S14 =  NetCDFFile(directory.format(S1_name,S1_name + mask4.format(domain)))
S04 =  NetCDFFile(directory.format(S0_name,S0_name + mask4.format(domain)))




# R1, 0: Rad, AP On
# R0, 1: Rad, AP Off
# S1, 3: No rad, AP On
# S0, 4: No rad, AP Off


titles = ['R1','R0','RD0','S1','S0']
titles2 = ['\mathrm{R_1A_1}','\mathrm{R_1A_0}', '\mathrm{R_0A_1}', '\mathrm{R_0A_0}']








[print (v) for v in R11.variables]





start = -27 #int(7200/300)
end = None



j = 0



heights = '{} m above surface'



def plot_aerosol(var,height=0):
    
    var1 = 'u'
    var2 = 'v'
    var3 = 'w'
    var4 = 'theta'
    j = height

    
    u_arrs = [read_var(base1,var1)[:,:,:],read_var(comp1,var1)[:,:,:]]
    v_arrs = [read_var(base1,var2)[:,:,:],read_var(comp1,var2)[:,:,:]]
    w_arrs = [read_var(base1,var3)[:,:,:],read_var(comp1,var3)[:,:,:]]
    T_arrs = [read_var(base1,var4)[:,:,:],read_var(comp1,var4)[:,:,:]]
    
    print (u_arrs[0].shape,u_arrs[1].shape)

    uv1 = np.hypot(u_arrs[0][:,j],v_arrs[0][:,j])
    uv2 = np.hypot(u_arrs[1][:,j],v_arrs[1][:,j])

    var = 'salsa_Ntot'
    print (var.split('_'))
    Vars = [read_var(base2,var)[:,:,:],read_var(comp2,var)[:,:,:]]
    n=14
    nc = 3
 
    X, Y = np.meshgrid(np.arange(0,461,n),np.arange(0,461,n))
    fig, axes = plt.subplots(ncols=nc,figsize=(14,6),constrained_layout=True)


    run = 0
    
    cmap = cm.roma_r
    cmap.set_bad('dimgrey')


    timestrs = [titles[base_i], titles[comp_i]]


    for i in range(2):
        u = u_arrs[i][:,j]
        v = v_arrs[i][:,j]

        q = axes[i].quiver(X,Y,u[::n,::n],v[::n,::n],scale=50)


        axes[i].set_title('{}'.format(timestrs[i]),fontsize=14)
        axes[i].set_xlabel('x(m)',fontsize=12)
    
    mean_base = Vars[0][:,j]
    mean_comp = Vars[1][:,j]

    
    std_base = np.std(np.mean(uv1,axis=(1,2)),axis=0)
    std_comp = np.std(np.mean(uv2,axis=(1,2)),axis=0)
    
    
    print ('{:.2e} +- {:.2e} & {:.2e} +- {:.2e} ({:.2f}\%) &'.format(np.nanmean(mean_base),std_base,np.nanmean(mean_comp),std_comp,
                                    np.nanmean((mean_comp-mean_base)/mean_base*100)))

    
    minv = np.floor(np.log10(np.min([mean_comp[mean_comp>0],mean_base[mean_base>0]])))
    maxv = np.ceil(np.log10(np.max([mean_comp,mean_base])))
        
    surf = axes[0].imshow(mean_base,cmap=cmap,norm=mpl.colors.LogNorm(vmin=10**minv,vmax=5*10**(maxv-1)),origin='lower')
    surf = axes[1].imshow(mean_comp,cmap=cmap,norm=mpl.colors.LogNorm(vmin=10**minv,vmax=5*10**(maxv-1)),origin='lower')




    axes[0].set_ylabel('y(m)',fontsize=12)    
    
    
    cbar = fig.colorbar(surf, ax=axes[:-1].ravel().tolist(),orientation='horizontal',aspect=30)
    cbar.set_label('{} ({})'.format(var.split('_')[-1],units),fontsize=14)
    

    delta = ((mean_comp-mean_base)/mean_base)*100
    delta[np.abs(delta)<0.001] = np.nan
    

    lower  = [delta.min(), -150, delta.max()]
    higher = [delta.min(), 150, delta.max()]

    
    cmap2 = cm.vik
    cmap2.set_bad('dimgrey')
    axes[-1].set_facecolor('dimgrey')


    bounds = np.arange(-150, 151, 10)

    p = axes[-1].contourf(delta, levels=bounds, cmap=cmap2, vmin=-151, vmax=151, extend='both')
    axes[-1].contourf(delta, levels=higher, hatches=['','++++'], alpha=0.0)
    axes[-1].contourf(delta, levels=lower,  hatches=['----',''], alpha=0.0)


    

    axes[-1].set_xlabel('x(m)',fontsize=12)  
    axes[-1].set_title(r'$\Delta${} ({} - {})'.format(var.split('_')[-1],timestrs[0],timestrs[-1]),fontsize=14)
    axes[-1].set_aspect(1)

    cbar2 = fig.colorbar(p,ax=[axes[-1]],orientation='horizontal',aspect=15)
    cbar2.set_label(r'$\Delta$(%)',fontsize=14)

    plt.savefig('{}_mask4_4m_{}vs{}_NEW.png'.format(var.split('_')[-1],timestrs[0],timestrs[1]),dpi=300)
    plt.savefig('{}_mask4_4m_{}vs{}_NEW.pdf'.format(var.split('_')[-1],timestrs[0],timestrs[1]),dpi=300)


def plot_aerosol_4x4(var,height=1):

    txts = ['a)', 'b)', 'c)', 'd)']
    tcolors = ['black','black','black','black']

    xp = S01.variables['x'][:]
    yp = S01.variables['y'][:]
    zp = S01.variables['ku_above_surf'][:]

    Np = S01.variables['N_UTM'][:]
    Ep = S01.variables['E_UTM'][:]
    
    palm_proj = pyproj.Proj( 'epsg:3879' )
    lons_palm, lats_palm = palm_proj( Ep, Np, inverse=True )
    
    
    LON, LAT = np.meshgrid( lons_palm, lats_palm )


    print (zp)
    var1 = 'u'
    var2 = 'v'
    var3 = 'w'
    var4 = 'theta'
    j = height

    
    u_arrs = [read_var(S01,var1)[j,:,:],read_var(S11,var1)[j,:,:],read_var(R01,var1)[j,:,:],read_var(R11,var1)[j,:,:]]
    v_arrs = [read_var(S01,var2)[j,:,:],read_var(S11,var2)[j,:,:],read_var(R01,var2)[j,:,:],read_var(R11,var2)[j,:,:]]
    w_arrs = [read_var(S01,var3)[j,:,:],read_var(S11,var3)[j,:,:],read_var(R01,var3)[j,:,:],read_var(R11,var3)[j,:,:]]
    T_arrs = [read_var(S01,var4)[j,:,:],read_var(S11,var4)[j,:,:],read_var(R01,var4)[j,:,:],read_var(R11,var4)[j,:,:]]
    
    


    var = 'salsa_Ntot'
    print (var.split('_'))
    Vars = [read_var(S04,var)[:,:,:],read_var(S14,var)[:,:,:], read_var(R04,var)[:,:,:], read_var(R14,var)[:,:,:]]

    n=14
    nc = 3
 
    X, Y = np.meshgrid(np.arange(0,461,n),np.arange(0,461,n))
    fig, ax = plt.subplots(ncols=2,nrows=2,sharey='row',sharex='col',figsize=(14,12),constrained_layout=True)

    axes = ax.ravel().tolist()

    

    run = 0
    
    cmap = cm.roma_r
    cmap.set_bad('dimgrey')
    
    cmap2 = cm.vik
    cmap2.set_bad('dimgrey')


    titles = [titles2[-1],'{} - {}'.format(titles2[-2],titles2[-1]),'{} - {}'.format(titles2[1],titles2[-1]),'{} - {}'.format(titles2[0],titles2[-1])]


    base = Vars[0][j]*1E-6

    minv = np.floor(np.log10(np.min(base[base>0])))
    maxv = np.ceil(np.log10(np.max(base)))

    cmc = make_colormap(8)

    for i in range(4):
        u = u_arrs[i]
        v = v_arrs[i]



        axes[i].set_title(r'${}$'.format(titles[i]),fontsize=18)


    
        if (i==0):    
            Ntot = base
            plot1 = axes[0].imshow(base,cmap=cmap,norm=mpl.colors.LogNorm(vmin=10**minv,vmax=5*10**(maxv-1)),origin='lower',extent=[LON[0,0], LON[0,-1], LAT[0,0],LAT[-1,0]])
            cbar1 = fig.colorbar(plot1, ax=axes[0],orientation='vertical',aspect=30,shrink=0.9)

            cbar1.set_label(r'$\mathrm{N_{tot,4m} (cm^{-3})}$',fontsize=18)
            cbar1.ax.tick_params(labelsize=16) 

            
            
        elif (i==1):
            Ntot = Vars[i][j]*1E-6
            print (np.mean((Ntot-base)/base*100.))
            
            axes[i].set_facecolor('dimgrey')
            



            delta = ((Ntot-base)/base)*100
            delta[np.abs(delta)<0.001] = np.nan
    

            lower  = [delta.min(), -100, delta.max()]
            higher = [delta.min(), 100, delta.max()]
  

            bounds = np.arange(-100, 101, 10)
            
            plot2 = axes[i].contourf(LON,LAT,delta, levels=bounds, cmap=cmap2, vmin=-101, vmax=101, extend='both')
            axes[i].contourf(LON,LAT,delta, levels=higher, hatches=['','++++'], alpha=0.0)
            axes[i].contourf(LON,LAT,delta, levels=lower,  hatches=['----',''], alpha=0.0)


        else:
            Ntot = Vars[i][j]*1E-6
            print (np.mean((Ntot-base)/base*100.))

            axes[i].set_facecolor('dimgrey')



            delta = ((Ntot-base)/base)*100
            delta[np.abs(delta)<0.001] = np.nan
    

    
            plot2 = axes[i].contourf(LON,LAT,delta, levels=bounds, cmap=cmap2, vmin=-101, vmax=101, extend='both')
            axes[i].contourf(LON,LAT,delta, levels=higher, hatches=['','++++'], alpha=0.0)
            axes[i].contourf(LON,LAT,delta, levels=lower,  hatches=['----',''], alpha=0.0)

        axes[i].tick_params(axis='both', labelsize=18)

        axes[i].text(0.88, 0.9, txts[i], color=tcolors[i],transform=axes[i].transAxes,
                fontsize=18,bbox=dict(facecolor='0.8', edgecolor='none', pad=3.0))
        q = axes[i].quiver(LON[::n,::n],LAT[::n,::n],u[::n,::n],v[::n,::n],scale=50)
        axes[i].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
        axes[i].set_yticks( [ 60.196, 60.197, 60.198, 60.199])
        axes[i].set_xticks( [ 24.948, 24.950, 24.952, 24.954])

        axes[i].plot(24.952011, 60.196433,'k*',markersize=14,linewidth=1, markerfacecolor=cmc(0.5), alpha=1, label='SR1')



        axes[i].set_aspect(2)
        
        print ('mean: {}'.format(np.nanmean(Ntot)))
        print ('max: {}'.format(np.nanmax(Ntot)))
        print ('min: {}'.format(np.nanmin(Ntot)))



       
    u_arrs,v_arrs,w_arrs,T_arrs = (None,None,None,None)

    cbar2 = fig.colorbar(plot2,ax=axes[1:],orientation='vertical',aspect=30)
    cbar2.set_label(r'$\Delta$(%)',fontsize=18,rotation=0)
    cbar2.ax.tick_params(labelsize=16) 

    plt.quiverkey(q,0.475,0.49,2,r'2$\frac{m}{s}$',coordinates='figure',fontproperties={'size':18})

    [axes[i].set_xlabel('Longitude ($^\circ$)',fontsize=18) for i in range(2,4)]
    [axes[i].set_ylabel('Latitude ($^\circ$)',fontsize=18) for i in range(0,4,2)]
    plt.savefig('Ntot_comparison_4x4_{}m_NEW.png'.format(int(zp[height])),dpi=300)
    plt.savefig('Ntot_comparison_4x4_{}m_NEW.pdf'.format(int(zp[height])),dpi=300)



    

def plot_aerosol_diffs():

    timestrs = [titles[base_i], titles[comp_i]]

    fig, axes = plt.subplots(nrows=2,ncols=5,figsize=(20,10),constrained_layout=True)

    for i,ax in enumerate(axes.flatten()):
    
        var = f'salsa_N_bin{i+1}'
        
        Vars = [read_var(base2,var)[:,:,:],read_var(comp2,var)[:,:,:]]
    
        height = 1
        
        comp = Vars[0][:,height]
        base = Vars[1][:,height]
        

        delta = ((comp-base)/base)*100
        delta[np.abs(delta)<0.1] = np.nan
        

        lower = [delta.min(), -150, delta.max()]
        higher = [delta.min(), 150, delta.max()]
      

        
        cmap2 = cm.vikO
        cmap2.set_bad('dimgrey')
        ax.set_facecolor('dimgrey')
    
    
        bounds = np.arange(-150, 151, 10)
    
        p = ax.contourf(delta, levels=bounds, cmap=cmap2, vmin=-150, vmax=150, extend='both')
        ax.contourf(delta, levels=higher, hatches=['','++++'], alpha=0.0)
        ax.contourf(delta, levels=lower,  hatches=['----',''], alpha=0.0)

        ax.set_title(r'{}'.format(var.split('_')[-1],fontsize=14))
    plt.suptitle('{} vs. {}'.format(titles[base_i],titles[comp_i]),fontsize=14)


    cbar2 = fig.colorbar(p,ax=[axes.flatten()],orientation='horizontal',aspect=30)
    cbar2.set_label(r'$\Delta$(%)',fontsize=14)
    plt.savefig('bins_change_4m_{}_vs_{}_NEW.png'.format(timestrs[0],timestrs[1]),dpi=300)

def t_test(x,y,alternative='both-sided'):
    _, double_p = stats.ttest_ind(x,y,equal_var = False)
    if alternative == 'both-sided':
        pval = double_p
    elif alternative == 'greater':
        if np.mean(x) > np.mean(y):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    elif alternative == 'less':
        if np.mean(x) < np.mean(y):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    return pval



plot_aerosol_4x4('salsa_Ntot',height=1)

