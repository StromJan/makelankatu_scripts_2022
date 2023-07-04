#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:39:20 2021

@author: stromjan
"""

from netCDF4 import Dataset as NetCDFFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats.mstats import gmean
from cmcrameri import cm



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

    return out[:]


rev = ''



directory = '/media/stromjan/Work_Drive/Puhti/JOBS/{}/OUTPUT/{}'
fixed = '_fixed_flow'

mask6 = '_av_masked_{}_M06.COMBINED.nc'
mask7 = '_av_masked_{}_M07.COMBINED.nc'


R1_name = '0609_morning_on_processes'+fixed
R0_name = '0609_morning_on_noprocesses'+fixed



S1_name = '0609_morning_off_processes'+fixed
S0_name = '0609_morning_off_noprocesses'+fixed

version = ''

local = True



var = 'salsa_PM2.5'


domain = 'N03'

cmap = mpl.cm.Spectral_r



anim = False




R16 =  NetCDFFile(directory.format(R1_name,R1_name + mask6.format(domain)))
R06 =  NetCDFFile(directory.format(R0_name,R0_name + mask6.format(domain)))



S16 =  NetCDFFile(directory.format(S1_name,S1_name + mask6.format(domain)))
S06 =  NetCDFFile(directory.format(S0_name,S0_name + mask6.format(domain)))

R17 =  NetCDFFile(directory.format(R1_name,R1_name + mask7.format(domain)))
R07 =  NetCDFFile(directory.format(R0_name,R0_name + mask7.format(domain)))



S17 =  NetCDFFile(directory.format(S1_name,S1_name + mask7.format(domain)))
S07 =  NetCDFFile(directory.format(S0_name,S0_name + mask7.format(domain)))


drone1 = [45.47591778, 46.14977802, 40.37741437, 33.33435294, 30.4492496,  27.3916169,
 23.52679863, 23.16734704, 21.34234044, 20.72332858, 21.16617566, 19.98488572]

drone1err = [[17.65292589, 22.48436953, 19.69717215, 14.18117584, 12.62898706,  9.50130716,
   7.48598314,  7.7514853,   6.32758553,  6.07611525,  6.31248257,  5.06173111],
 [29.40774873, 45.41599946, 41.40346186, 26.99317998, 22.8184392,  15.14417773,
  11.84811736, 12.21879346,  9.32377039,  9.28602296,  9.86984818,  7.41915431]]

drone2 = [42.01135079, 37.57642141, 30.43330849, 29.18737792, 25.80105184, 23.53110114,
 21.07007966, 19.22500888, 18.35531885, 17.51625616, 16.93852572, 16.77280518]

drone2err =  [[17.83621687, 17.76862311, 13.7681529,  12.49489243, 10.12487379,  8.04164835,
   6.12008302,  4.70604538,  4.15818738,  3.39383884,  2.06852962,  1.90558694],
 [33.72674006, 37.65073604, 26.23011761, 22.43473209, 17.09394884, 13.11566027,
   8.97269625,  6.2377628,   5.41562736,  4.26206732,  2.36252959,  2.1725482 ]]





[print (v) for v in R16.variables]


start = -24 #int(7200/300)
end = None








j = 0





titles = ['\mathrm{R_1A_1}','\mathrm{R_1A_0}', '\mathrm{R_0A_1}','\mathrm{R_0A_0}']



heights = '{} m above surface'


def plot_aerosol_profiles():

    
    plots = ['salsa_Ntot','salsa_N_UFP','salsa_PM2.5','salsa_LDSA']
    plotS0 = ['salsa_s_SO4','salsa_s_OC','salsa_s_BC','salsa_s_NO','salsa_s_NH']
    txts = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)']
    unitlabs = [r' ($\mathrm{cm^{-3}}$)', r' ($\mathrm{cm^{-3}}$)', r' ($\mathrm{g~cm^{-3}}$)', r' ($\mathrm{\mu m^2cm^{-3}}$)']
    fig, axes = plt.subplots(ncols=2,nrows=4,figsize=(8,16),constrained_layout=True)

    y = np.arange(120)

    for r in range(4):
        var = plots[r]
        lines = ['r-','r--','b-','b--']


        
        Vars  = np.asarray([gmean(gmean(gmean(read_var(R16,var)[start:end,1:,:,:], axis=0), axis=1), axis=1),gmean(gmean(gmean(read_var(R06,var)[start:end,1:,:,:], axis=0), axis=1), axis=1),
                 gmean(gmean(gmean(read_var(S16,var)[start:end,1:,:,:], axis=0), axis=1), axis=1),gmean(gmean(gmean(read_var(S06,var)[start:end,1:,:,:], axis=0), axis=1), axis=1)])
        VarS0 = np.asarray([gmean(gmean(gmean(read_var(R17,var)[start:end,1:,:,:], axis=0), axis=1), axis=1),gmean(gmean(gmean(read_var(R07,var)[start:end,1:,:,:], axis=0), axis=1), axis=1),
                 gmean(gmean(gmean(read_var(S17,var)[start:end,1:,:,:], axis=0), axis=1), axis=1),gmean(gmean(gmean(read_var(S07,var)[start:end,1:,:,:], axis=0), axis=1), axis=1)])
        

        

    


        if (r<2):
            Vars = Vars/1E6
            VarS0 = VarS0/1E6
        elif (r==2):
            Vars = Vars/1E3
            VarS0 = VarS0/1E3
        
        
        regions = [Vars,VarS0]
        minv = np.floor(np.log10(np.min([Vars[Vars>0],VarS0[VarS0>0]])))
        maxv = np.ceil(np.log10(np.max([Vars,VarS0])))
    
        for i,data in enumerate(regions):
         
            if (var=='salsa_s_NO'):
                axes[r,i].set_xlim(8E-12,1E-10)
        
        
            cmap = cm.romaO_r
            cmap.set_bad('dimgrey')
            
            
            
            
            for j in range(len(titles)):
        
                axes[r,i].plot(data[j],y,lines[j],lw=2,label=r'${}$'.format(titles[j]))
                


        
            axes[r,i].set_xscale('log')
            axes[r,i].set_xlim(2*10**minv,10**maxv)
            axes[r,i].set_ylim(0,60)

    
            axes[r,i].grid(True)
    
    
            axes[r,i].set_xlabel(var.split('_')[-1]+unitlabs[r],fontsize=18)

    
        axes[r,0].set_ylabel('z (m)',fontsize=18)
    axes[0,0].set_title('Supersite (SR1)',fontsize=18)
    axes[0,1].set_title('Opposite supersite (SR2)',fontsize=18)
    
    

    axes[-1,0].errorbar(drone1, np.arange(4,50,4), xerr=drone1err, marker='s', capsize=2, color='g',label='Drone')
    axes[-1,1].errorbar(drone2, np.arange(4,50,4), xerr=drone2err, marker='s', capsize=2, color='g',label='Drone')
    axes[-1,1].legend(loc='center right',prop={'size': 14})

    for i,ax in enumerate(axes.flatten()):

        ax.text(0.88, 0.9, txts[i],transform=ax.transAxes,
                    fontsize=18,bbox=dict(facecolor='0.8', edgecolor='none', pad=3.0))
        ax.get_xaxis().set_tick_params(which='major', labelsize=18)
        ax.get_yaxis().set_tick_params(which='major', labelsize=18)


    plt.savefig('/home/stromjan/Output/Profiles1_all_runs_NEW.pdf',dpi=300)
    plt.savefig('/home/stromjan/Output/Profiles1_all_runs_NEW.png',dpi=300)




plot_aerosol_profiles()





