#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:35:12 2021

@author: stromjan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:05:36 2020

@author: stromjan
"""
#import os as sys
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as NetCDFFile
from scipy.signal import detrend
import seaborn as sns
from matplotlib.legend_handler import HandlerBase
import matplotlib.patches as mpatches
from useful_funcs import *
import warnings
from scipy import ndimage, misc,stats
import matplotlib as mpl
import os
os.chdir("/home/stromjan/Output/")

class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                           linestyle=orig_handle[1], color='k')
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], 
                           color=orig_handle[0])
        return [l1, l2]


def calculate_p(a1,a2,dependent=False):
    return stats.ttest_ind(a1,a2,nan_policy='propagate',equal_var=dependent)[-1]


warnings.filterwarnings("ignore", category=DeprecationWarning) 
#just for nicer plot cycle colors
sns.set()
pal = sns.hls_palette(h=.9)
plt.style.use('default')
#-------------------------------


pr_variables = ['u_00', 'u_01', 'u_02', 'u_03', 'u*2_00', 'u*2_01', 'u*2_02', 'u*2_03', 'wu_00', 'wu_01',
                'wu_02', 'wu_03', 'w*u*_00', 'w*u*_01', 'w*u*_02', 'w*u*_03', 'w"u"_00', 'w"u"_01', 'w"u"_02',
                'w"u"_03', 'v_00', 'v_01', 'v_02', 'v_03', 'v*2_00', 'v*2_01', 'v*2_02', 'v*2_03', 'wv_00',
                'wv_01', 'wv_02', 'wv_03', 'w*v*_00', 'w*v*_01', 'w*v*_02', 'w*v*_03', 'w"v"_00', 'w"v"_01',
                'w"v"_02', 'w"v"_03', 'w_00', 'w_01', 'w_02', 'w_03', 'w*2_00', 'w*2_01', 'w*2_02', 'w*2_03',
                'rho_00', 'rho_01', 'rho_02', 'rho_03', 'theta_00', 'theta_01', 'theta_02', 'theta_03',
                'theta*2_00', 'theta*2_01', 'theta*2_02', 'theta*2_03', 'wtheta_00', 'wtheta_01', 'wtheta_02',
                'wtheta_03', 'w*theta*_00', 'w*theta*_01', 'w*theta*_02', 'w*theta*_03', 'w"theta"_00',
                'w"theta"_01', 'w"theta"_02', 'w"theta"_03', 'thetav_00', 'thetav_01', 'thetav_02', 'thetav_03',
                'wthetav_00', 'wthetav_01', 'wthetav_02', 'wthetav_03', 'w*thetav*_00', 'w*thetav*_01',
                'w*thetav*_02', 'w*thetav*_03', 'w"thetav"_00', 'w"thetav"_01', 'w"thetav"_02', 'w"thetav"_03',
                'w*u*u*:dz_00', 'w*u*u*:dz_01', 'w*u*u*:dz_02', 'w*u*u*:dz_03', 'w*p*:dz_00', 'w*p*:dz_01',
                'w*p*:dz_02', 'w*p*:dz_03', 'q_00', 'q_01', 'q_02', 'q_03', 'q*2_00', 'q*2_01', 'q*2_02',
                'q*2_03', 'e_00', 'e_01', 'e_02', 'e_03', 'e*_00', 'e*_01', 'e*_02', 'e*_03']

pr_variables_rad = ['u_00', 'u_01', 'u_02', 'u_03', 'u*2_00', 'u*2_01', 'u*2_02', 'u*2_03', 'wu_00', 'wu_01',
                    'wu_02', 'wu_03', 'w*u*_00', 'w*u*_01', 'w*u*_02', 'w*u*_03', 'w"u"_00', 'w"u"_01',
                    'w"u"_02', 'w"u"_03', 'v_00', 'v_01', 'v_02', 'v_03', 'v*2_00', 'v*2_01', 'v*2_02',
                    'v*2_03', 'wv_00', 'wv_01', 'wv_02', 'wv_03', 'w*v*_00', 'w*v*_01', 'w*v*_02', 'w*v*_03',
                    'w"v"_00', 'w"v"_01', 'w"v"_02', 'w"v"_03', 'w_00', 'w_01', 'w_02', 'w_03', 'w*2_00',
                    'w*2_01', 'w*2_02', 'w*2_03', 'rho_00', 'rho_01', 'rho_02', 'rho_03', 'm_soil_00',
                    'm_soil_01', 'm_soil_02', 'm_soil_03', 't_soil_00', 't_soil_01', 't_soil_02', 't_soil_03',
                    'theta_00', 'theta_01', 'theta_02', 'theta_03', 'theta*2_00', 'theta*2_01', 'theta*2_02',
                    'theta*2_03', 'wtheta_00', 'wtheta_01', 'wtheta_02', 'wtheta_03', 'w*theta*_00',
                    'w*theta*_01', 'w*theta*_02', 'w*theta*_03', 'w"theta"_00', 'w"theta"_01', 'w"theta"_02',
                    'w"theta"_03', 'thetav_00', 'thetav_01', 'thetav_02', 'thetav_03', 'wthetav_00',
                    'wthetav_01', 'wthetav_02', 'wthetav_03', 'w*thetav*_00', 'w*thetav*_01', 'w*thetav*_02',
                    'w*thetav*_03', 'w"thetav"_00', 'w"thetav"_01', 'w"thetav"_02', 'w"thetav"_03',
                    'w*u*u*:dz_00', 'w*u*u*:dz_01', 'w*u*u*:dz_02', 'w*u*u*:dz_03', 'w*p*:dz_00', 'w*p*:dz_01',
                    'w*p*:dz_02', 'w*p*:dz_03', 'rad_lw_in_00', 'rad_lw_in_01', 'rad_lw_in_02', 'rad_lw_in_03',
                    'rad_lw_out_00', 'rad_lw_out_01', 'rad_lw_out_02', 'rad_lw_out_03', 'rad_sw_in_00',
                    'rad_sw_in_01', 'rad_sw_in_02', 'rad_sw_in_03', 'rad_sw_out_00', 'rad_sw_out_01',
                    'rad_sw_out_02', 'rad_sw_out_03', 'q_00', 'q_01', 'q_02', 'q_03', 'q*2_00', 'q*2_01',
                    'q*2_02', 'q*2_03', 'e_00', 'e_01', 'e_02', 'e_03', 'e*_00', 'e*_01', 'e*_02', 'e*_03']

ts_variables = ['E_00', 'E_01', 'E_02', 'E_03', 'E*_00', 'E*_01', 'E*_02', 'E*_03', 'dt_00', 'dt_01', 'dt_02',
                'dt_03', 'us*_00', 'us*_01', 'us*_02', 'us*_03', 'th*_00', 'th*_01', 'th*_02', 'th*_03',
                'umax_00', 'umax_01', 'umax_02', 'umax_03', 'vmax_00', 'vmax_01', 'vmax_02', 'vmax_03',
                'wmax_00', 'wmax_01', 'wmax_02', 'wmax_03', 'div_new_00', 'div_new_01', 'div_new_02',
                'div_new_03', 'div_old_00', 'div_old_01', 'div_old_02', 'div_old_03', 'zi_wtheta_00',
                'zi_wtheta_01', 'zi_wtheta_02', 'zi_wtheta_03', 'zi_theta_00', 'zi_theta_01', 'zi_theta_02',
                'zi_theta_03', 'w*_00', 'w*_01', 'w*_02', 'w*_03', 'w"theta"0_00', 'w"theta"0_01',
                'w"theta"0_02', 'w"theta"0_03', 'w"theta"_00', 'w"theta"_01', 'w"theta"_02', 'w"theta"_03',
                'wtheta_00', 'wtheta_01', 'wtheta_02', 'wtheta_03', 'theta(0)_00', 'theta(0)_01', 'theta(0)_02',
                'theta(0)_03', 'theta(z_mo)_00', 'theta(z_mo)_01', 'theta(z_mo)_02', 'theta(z_mo)_03',
                'w"u"0_00', 'w"u"0_01', 'w"u"0_02', 'w"u"0_03', 'w"v"0_00', 'w"v"0_01', 'w"v"0_02', 'w"v"0_03',
                'w"q"0_00', 'w"q"0_01', 'w"q"0_02', 'w"q"0_03', 'ol_00', 'ol_01', 'ol_02', 'ol_03', 'q*_00',
                'q*_01', 'q*_02', 'q*_03', 'w"s"_00', 'w"s"_01', 'w"s"_02', 'w"s"_03', 's*_00', 's*_01',
                's*_02', 's*_03']

ts_variables_rad = ['E_00', 'E_01', 'E_02', 'E_03', 'E*_00', 'E*_01', 'E*_02', 'E*_03', 'dt_00', 'dt_01',
                    'dt_02', 'dt_03', 'us*_00', 'us*_01', 'us*_02', 'us*_03', 'th*_00', 'th*_01', 'th*_02',
                    'th*_03', 'umax_00', 'umax_01', 'umax_02', 'umax_03', 'vmax_00', 'vmax_01', 'vmax_02',
                    'vmax_03', 'wmax_00', 'wmax_01', 'wmax_02', 'wmax_03', 'div_new_00', 'div_new_01',
                    'div_new_02', 'div_new_03', 'div_old_00', 'div_old_01', 'div_old_02', 'div_old_03',
                    'zi_wtheta_00', 'zi_wtheta_01', 'zi_wtheta_02', 'zi_wtheta_03', 'zi_theta_00',
                    'zi_theta_01', 'zi_theta_02', 'zi_theta_03', 'w*_00', 'w*_01', 'w*_02', 'w*_03',
                    'w"theta"0_00', 'w"theta"0_01', 'w"theta"0_02', 'w"theta"0_03', 'w"theta"_00',
                    'w"theta"_01', 'w"theta"_02', 'w"theta"_03', 'wtheta_00', 'wtheta_01', 'wtheta_02',
                    'wtheta_03', 'theta(0)_00', 'theta(0)_01', 'theta(0)_02', 'theta(0)_03', 'theta(z_mo)_00',
                    'theta(z_mo)_01', 'theta(z_mo)_02', 'theta(z_mo)_03', 'w"u"0_00', 'w"u"0_01', 'w"u"0_02',
                    'w"u"0_03', 'w"v"0_00', 'w"v"0_01', 'w"v"0_02', 'w"v"0_03', 'w"q"0_00', 'w"q"0_01',
                    'w"q"0_02', 'w"q"0_03', 'ol_00', 'ol_01', 'ol_02', 'ol_03', 'q*_00', 'q*_01', 'q*_02',
                    'q*_03', 'w"s"_00', 'w"s"_01', 'w"s"_02', 'w"s"_03', 's*_00', 's*_01', 's*_02', 's*_03',
                    'ghf_00', 'ghf_01', 'ghf_02', 'ghf_03', 'qsws_liq_00', 'qsws_liq_01', 'qsws_liq_02',
                    'qsws_liq_03', 'qsws_soil_00', 'qsws_soil_01', 'qsws_soil_02', 'qsws_soil_03',
                    'qsws_veg_00', 'qsws_veg_01', 'qsws_veg_02', 'qsws_veg_03', 'r_a_00', 'r_a_01', 'r_a_02',
                    'r_a_03', 'r_s_00', 'r_s_01', 'r_s_02', 'r_s_03', 'rad_net_00', 'rad_net_01', 'rad_net_02',
                    'rad_net_03', 'rad_lw_in_00', 'rad_lw_in_01', 'rad_lw_in_02', 'rad_lw_in_03',
                    'rad_lw_out_00', 'rad_lw_out_01', 'rad_lw_out_02', 'rad_lw_out_03', 'rad_sw_in_00',
                    'rad_sw_in_01', 'rad_sw_in_02', 'rad_sw_in_03', 'rad_sw_out_00', 'rad_sw_out_01',
                    'rad_sw_out_02', 'rad_sw_out_03', 'rrtm_aldif_00', 'rrtm_aldif_01', 'rrtm_aldif_02',
                    'rrtm_aldif_03', 'rrtm_aldir_00', 'rrtm_aldir_01', 'rrtm_aldir_02', 'rrtm_aldir_03',
                    'rrtm_asdif_00', 'rrtm_asdif_01', 'rrtm_asdif_02', 'rrtm_asdif_03', 'rrtm_asdir_00',
                    'rrtm_asdir_01', 'rrtm_asdir_02', 'rrtm_asdir_03']


g = 9.81
cp = 1004.
rho = 1.1724366
k = 0.41
zh = 13.2763546

#groundval = grid point from which vertical axis is spliced to only get values above ground
def read_pr_var(ds,var,runn=0,h=None,detrending=False,groundval=0):
  global idx

  v = ds.variables[var]
  print (var,v.units)
  v = v[4500:,groundval:]
    
  print (groundval)

  if (h is None):
      if (detrending):
          v = detrend(v,axis=0,type='linear')
          return v
      else:

          return v


  else:
      return np.asarray(v[:,h])
  
def read_ts_var(ds,var):
    v = ds.variables[var]
    return np.asarray(v)


def read_wind_temp_profile(ds,area,i,groundval):
    u, v, w, t         = (ds.variables['u'+area][4500:,groundval:],ds.variables['v'+area][4500:,groundval:],ds.variables['w'+area][4500:,groundval:],ds.variables['theta'+area][4500:,groundval:])
    up, vp, wp, tp     = (detrend(u,axis=0,type='linear'),detrend(v,axis=0,type='linear'),detrend(w,axis=0,type='linear'),detrend(t,axis=0,type='linear'))
    stu, stv, stw, stt = (np.std(u,axis=0),np.std(v,axis=0),np.std(w,axis=0),np.std(t,axis=0))
    return (u,v,w,t,up,vp,wp,tp,stu,stv,stw,stt)



def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)
  


plot = 'pr'
domain = 'child'


nests = {'root': ('',0), 'parent': ('_N02',1), 'child': ('_N03',2)}
colors = ['b', 'r']
colors2 = ['--r','--b']
resolutions = [6,3,1] #vertical resolution of the domains (root,parent,child)




directory = '/media/stromjan/Work_Drive/Puhti/JOBS/{}/OUTPUT/{}'

directory2 = r'/home/stromjan/palm/current_version/JOBS/urban_environment_4734/OUTPUT/{}'


prof = '_pr_N03.COMBINED.nc'


fixed = '_fixed_flow'

R1_name = '0609_morning_on_processes' + fixed

S0_name = '0609_morning_off_noprocesses' + fixed

version = ''

local = True



var = 'u'
var2 = 'v'
var3 = 'w'
var4 = 'theta'

domain = 'N03'

cmap = mpl.cm.Spectral_r



anim = False


R11 =  NetCDFFile(directory.format(R1_name,R1_name + prof.format(domain)))

S01 =  NetCDFFile(directory.format(S0_name,S0_name + prof.format(domain)))





titles = ['S0','R1']


#nr: statistical region id, manual: True if perturbations calculated manually
        
region = ''
var_names = ['u','v','w','w*u*','w*v*','w*theta*','theta','w*u*u*:dz','w*p*:dz','e','e*','rad_lw_in','rad_lw_out']

var_names = [var + region for var in var_names]

ds = [S01,R11]



runs = ['$\mathrm{R_0A_0}$','$\mathrm{R_0A_1}$', '$\mathrm{R_1A_1}*$', '$\mathrm{R_1A_0}$','$\mathrm{R_1A_1}$']
ln = ['-b','-r']

#ground level height above sealevel for different statistical regions
#[SR1,SR2,SR3,whole domain]
region_heights = [3,24,24,20]


domains = ['Child average (SR0)', 'Supersite (SR1)', 'Opposite supersite (SR2)', 'Background (SR3)']
x_labels = ['Wind speed U (m/s)',r'Potential temperature $\Theta\ (K)$','TKE ($m^2/s^2$)']



plot = 'met'
rown = {'budget':2,'met':3}



fig, axes = plt.subplots(ncols=4,nrows=rown.get(plot),sharey=True,figsize=(12,18),constrained_layout=True)
rows = rown.get(plot)
means = []
profiles = []
#determining the starting point for y-axis    
for nr in range(4):
    print ('\n')
    
    winds = []
    thetas = []
    ess = []
    for i,data in enumerate(ds):
        ground = region_heights[nr]

    
    
        area = f'_0{nr}'
    
    
        u, v, w, theta, up, vp, wp, tp, stu, stv, stw, stt = read_wind_temp_profile(data,area,i,ground)

        
        pr1 = 25
        pr2 = 75




        
        
            




    
        
        y = np.arange(len(u[0]))
    
        
        if (plot=='budget'):
            rows = 2
            dudz = np.gradient(u,axis=1)
            dvdz = np.gradient(v,axis=1)
            
            
            x_labels = [r'Mechanical production ($m^2/s^3$)',r'Thermal production ($m^2/s^3$)']
            wu     = read_pr_var(data,'wu'+area,runn=i,groundval=ground)
            wv     = read_pr_var(data,'wv'+area,runn=i,groundval=ground)
            wtheta = read_pr_var(data,'wtheta'+area,runn=i,groundval=ground)

            if (i==0): 
                wtheta /= (cp*rho)
                wu/=rho
                wv/=rho
    
    

            wuv = -wu*dudz -wv*dvdz
            wtheta *= (g/theta)
            
            
            
            wuv25, wuv75 = (np.nanpercentile(wuv,25,axis=0),np.percentile(wuv,75,axis=0))
            wt25, wt75 = (np.nanpercentile(wtheta,25,axis=0),np.percentile(wtheta,75,axis=0))
            
            wuv = np.nanmean(wuv,axis=0)
            wtheta = np.nanmean(wtheta,axis=0)



            axes[0,nr].plot(wuv,y*resolutions[nests[domain][1]],colors[i],label='RRTMG {}'.format(runs[i]),lw=1)
            axes[1,nr].plot(wtheta,y*resolutions[nests[domain][1]],colors[i],label='RRTMG {}'.format(runs[i]),lw=1)
            
            axes[0,nr].fill_betweenx(y,wuv25,wuv75,color=colors[i],alpha=0.2)
            axes[1,nr].fill_betweenx(y,wt25,wt75,color=colors[i],alpha=0.2)

            axes[0,nr].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axes[1,nr].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            
            print(f'({titles[i]}, SR{nr}): \n mechanical: {np.mean(wuv)} \n thermal: {np.mean(wtheta)} \n')
            means.append((np.mean(wuv),np.mean(wtheta)))



            
        else:
                    
            
            
            u25 = np.percentile(np.hypot(u,v,w),pr1,axis=0)
            u75 = np.percentile(np.hypot(u,v,w),pr2,axis=0)
    
            t25 = np.percentile(theta,pr1,axis=0)
            t75 = np.percentile(theta,pr2,axis=0)
    

            dirh = 0
            Dir=mean_wdir(np.mod(180+np.rad2deg(np.arctan2(u[:3600,dirh:], v[:3600,dirh:])),360))
            print (f'Mean wind direction (first hour): {titles[i]} SR{nr} {Dir}')
            
            Dir2=mean_wdir(np.mod(180+np.rad2deg(np.arctan2(u[3600:,dirh:], v[3600:,dirh:])),360))
            print (f'Mean wind direction (last hour):  {titles[i]} SR{nr} {Dir2}')

            Dir3=mean_wdir(np.mod(180+np.rad2deg(np.arctan2(u[:,dirh:], v[:,dirh:])),360))
            print (f'Mean wind direction (full run):  {titles[i]} SR{nr} {Dir3}')
    
    
    
            thetaplot = np.mean(theta,axis=0)

            uvw = np.hypot(u,v,w)

    
            uvwplot = np.mean(np.hypot(u,v,w),axis=0)

    
            
            e = read_pr_var(data,'e*'+area,i,detrending=False,groundval=ground)
            es = read_pr_var(data,'e'+area,i,detrending=False,groundval=ground)
            e += es

            e25, e75 = (np.percentile(e,25,axis=0),np.percentile(e,75,axis=0))
            eplot = np.mean(e,axis=0)



            rows = 3

            if (nr==-1):

             
                
                WD1 = np.mod(180+np.rad2deg(np.arctan2(u[:3600,dirh:], v[:3600,dirh:])),360)
                WD2 = np.mod(180+np.rad2deg(np.arctan2(u[3600:,dirh:], v[3600:,dirh:])),360)
                WD = np.mod(180+np.rad2deg(np.arctan2(u[:,dirh:], v[:,dirh:])),360)
                
                
                WD1pr1, WD1pr2 = (np.percentile(WD1,pr1,axis=0),np.percentile(WD1,pr2,axis=0))
                WD2pr1, WD2pr2 = (np.percentile(WD2,pr1,axis=0),np.percentile(WD2,pr2,axis=0))
                WDpr1, WDpr2 = (np.percentile(WD,pr1,axis=0),np.percentile(WD,pr2,axis=0))
                
                
                WD1 = mean_wdir(WD1,axis=0)
                WD2 = mean_wdir(WD2,axis=0)
                WD = mean_wdir(WD,axis=0)



                axes2[0].plot(WD1,y[dirh:],colors[i],label=titles[i]+r' ({:3.1f}$^\circ$)'.format(np.mean(WD1)),lw=2)
                axes2[1].plot(WD2,y[dirh:],colors[i],label=titles[i]+r' ({:3.1f}$^\circ$)'.format(np.mean(WD2)),lw=2)
                axes2[2].plot(WD,y[dirh:],colors[i],label=titles[i]+r' ({:3.1f}$^\circ$)'.format(np.mean(WD)),lw=2)
    
    
                axes2[0].fill_betweenx(y[dirh:],WD1pr1,WD1pr2,color=colors[i],alpha=0.2)
                axes2[1].fill_betweenx(y[dirh:],WD2pr1,WD2pr2,color=colors[i],alpha=0.2)
                axes2[2].fill_betweenx(y[dirh:],WDpr1,WDpr2,color=colors[i],alpha=0.2)

    
            axes[0,nr].plot(uvwplot,y,colors[i],label=titles[i],lw=2)
            axes[1,nr].plot(thetaplot,y,colors[i],label=titles[i],lw=2)
            axes[2,nr].plot(eplot[:-2],y[:-2],colors[i],label=titles[i],lw=2)
            

            windm = np.mean(uvwplot[:])
            thetam = np.mean(thetaplot[:])
            em = np.mean(eplot[:])
            
            print(f'({titles[i]}, SR{nr}): \n wind: {windm} \n theta: {thetam} \n TKE: {em} \n',
                  f'Theta closest to ground: {np.mean(theta,axis=0)[0]} Kelvin\n')
            
            means.append((windm,thetam,em))
            

            

            

            axes[0,nr].fill_betweenx(y,u25,u75,color=colors[i],alpha=0.2)
            axes[1,nr].fill_betweenx(y,t25,t75,color=colors[i],alpha=0.2)
            axes[2,nr].fill_betweenx(y[:-2],e25[:-2],e75[:-2],color=colors[i],alpha=0.2)









[axes[j,0].set_ylabel('Height above surface (m)',fontsize=16) for j in range(rows)]
[axes[0,j].set_title(domains[j],fontsize=16) for j in range(4)]



fig.text(0.52, 0.665,  x_labels[0], ha='center',fontsize=14)
fig.text(0.52, 0.335, x_labels[1], ha='center',fontsize=14)
fig.text(0.52, 0.015, x_labels[2], ha='center',fontsize=14)

for j in range(rows):
    x_labels[j] = ' '
    axes[j,0].set_xlabel(x_labels[j],fontsize=16)
    axes[j,1].set_xlabel(x_labels[j],fontsize=16)
    axes[j,2].set_xlabel(x_labels[j],fontsize=16)
    axes[j,3].set_xlabel(x_labels[j],fontsize=16)

    axes[j,0].xaxis.labelpad=15
    axes[j,1].xaxis.labelpad=15
    axes[j,2].xaxis.labelpad=15
    axes[j,3].xaxis.labelpad=15






red_patch = mpatches.Patch(color='r', label='The red data',alpha=0.2)
blue_patch = mpatches.Patch(color='b', label='The blue data',alpha=0.2)


handles, labels = axes[0,0].get_legend_handles_labels()


fig.suptitle(' ')

leg = fig.legend(bbox_to_anchor=(0.02,0.665,1.,0.),handles=handles,labels=labels,loc='upper center',ncol=4,fontsize=12,facecolor='grey',framealpha=0.2,columnspacing=1.5)


for ax in (axes.flatten()):
    ax.grid(True)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick2 in ax.yaxis.get_major_ticks():
        tick2.label.set_fontsize(12)


means = np.asarray(means)




print (np.asarray(profiles).shape)

#
if (plot=='met'):
    ar1 = (means[0]-means[1])/means[1]*100
    ar2 = means[0+2]-means[1+2]
    ar3 = means[0+4]-means[1+4]

    print (('SR0 & {} & {} & {}').format(ar1[0],ar1[1],ar1[2]))
    fig.savefig('Profiles_met_Large.pdf',dpi=250)


elif (plot=='budget'):
    ar1 = (means[0]-means[1])/means[1]*100
    fig.savefig('Profiles_budget_Large.png',dpi=250)



    print (('SR0 & {} & {}').format(ar1[0],ar1[1]))











