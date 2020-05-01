#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 05:22:35 2019

@author: altsai
"""

import os
import sys
import shutil
import numpy as np
import csv
import time
import math
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord  # High-level coordinates
#from astropy.coordinates import ICRS, Galactic, FK4, FK5 # Low-level frames
#from astropy.coordinates import Angle, Latitude, Longitude  # Angles
#from astropy.coordinates import match_coordinates_sky
from astropy.table import Table
from photutils import CircularAperture
from photutils import SkyCircularAperture
from photutils import aperture_photometry
from photutils import CircularAnnulus
from photutils import SkyCircularAnnulus
# https://photutils.readthedocs.io/en/stable/aperture.html
#from phot import aperphot
# http://www.mit.edu/~iancross/python/phot.html

import matplotlib.pyplot as plt
import matplotlib.axes as ax
from astropy.io import fits
from astropy.wcs import WCS
#from photutils import DAOStarFinder
#from astropy.stats import mad_std
# https://photutils.readthedocs.io/en/stable/getting_started.html

from numpy.polynomial.polynomial import polyfit
from astropy.stats import sigma_clipped_stats
from photutils.psf import IterativelySubtractedPSFPhotometry
from statistics import mode
from astropy.visualization import simple_norm
from photutils.utils import calc_total_error

'''
# 3C345-20190714@135841-R_Astrodon_2018_calib.fits
#fits_root=input('Please input the file name of the fitsimage: ').split('.',-1)[0].split('_calib',-1)[0]
fits_root='3C345-20190822@130653-R_Astrodon_2018'
fits_calib=fits_root+'_calib.fits'
fits_ori=fits_root+'.fts'
#print(fits_root)
#print(fits_calib)
#print(fits_ori)

date=fits_root.split('@',-1)[0].split('-',-1)[-1]
print(date)
print(type(date))
yearmonth=date[0:6]
'''
#file_info='gasp_target_fitsheader_info_slt201907.txt'
file_info='gasp_target_fitsheader_info_slt2019.txt'
#file_info='gasp_target_fitsheader_info_slt'+date+'.txt'
#file_info='gasp_target_fitsheader_info_slt'+year+month+'.txt'
print('... will read '+file_info+' ...')
df_info=pd.read_csv(file_info,delimiter='|')
#print(df_info)

#obj_name='3C345'
#obj_name='ES2344+514'
#obj_name='L-Lacertae'
obj_name=sys.argv[1]


xfwhm=3

dir_obj='Rmag_InstMag/circle/'+obj_name+'/circle_r=fwhmx'+str(xfwhm)+'/'
if os.path.exists(dir_obj):
    shutil.rmtree(dir_obj)
os.makedirs(dir_obj,exist_ok=True)

dir_refstar='RefStar/'
file_refstar='gasp_refStar_radec.txt'


df_refstar=pd.read_csv(file_refstar,sep='|')
#print(df_refstar)
idx_refstar=df_refstar[df_refstar['ObjectName']==obj_name].index.tolist()
n_refstar=len(idx_refstar)
print('number of reference stars : ',n_refstar)
ra_deg=np.array([0.]*n_refstar)
dec_deg=np.array([0.]*n_refstar)
#ra_pix=np.array([0.]*n_refstar)
#dec_pix=np.array([0.]*n_refstar)
radec_deg=np.array([[0.,0.]]*n_refstar)
#radec_pix=np.array([[0.,0.]]*n_refstar)
ra_hhmmss=['']*n_refstar
dec_ddmmss=['']*n_refstar

rmag=np.array([0.]*n_refstar)
rmag_err=np.array([0.]*n_refstar)
mag_instrument=np.array([0.]*n_refstar)
#mag_err=np.array([0.]*n_refstar)
#snr=np.array([0.]*n_refstar)

print('-----------------------')




j1=0
for j2 in idx_refstar:
    ra_deg[j1]=df_refstar['RefStarRA_deg'][j2]
    dec_deg[j1]=df_refstar['RefStarDEC_deg'][j2]
#    print(ra_deg[j1],dec_deg[j1])
    radec_deg[j1]=[ra_deg[j1],dec_deg[j1]]
    print(j1,j2,radec_deg[j1])
    ra_hhmmss[j1]=df_refstar['RefStarRA_hhmmss'][j2]
    dec_ddmmss[j1]=df_refstar['RefStarDEC_ddmmss'][j2]
#    print(i,j1,radec_deg[j1],ra_hhmmss[j1],dec_ddmmss[j1])
    rmag[j1]=df_refstar['Rmag'][j2]
    rmag_err[j1]=df_refstar['Rmag_err'][j2]
    j1=j1+1




idx_fitsheader=df_info[df_info['Object']==obj_name].index
#print(idx_fitsheader)

#obj_name=df_info['Object'][idx_fitsheader]
fwhm=df_info['FWHM'][idx_fitsheader]
#print(fwhm)
ID=df_info['ID'][idx_fitsheader]
#JD=df_info['JD'][idx_fitsheader]

#sys.exit(0)

#position= SkyCoord([ICRS(ra=ra_deg*u.deg,dec=dec_deg*u.deg)])
positions= SkyCoord(ra_deg,dec_deg,unit=(u.hourangle,u.deg),frame='icrs')
#print(positions)
#print(positions.ra)
#print(positions.dec)


#sys.exit(0)
#=======================




fits_ori=df_info['Filename'][idx_fitsheader]
#fits_root=input('Please input the file name of the fitsimage: ').split('.',-1)[0].split('_calib',-1)[0]
#fits_root=['']*n_idx
#fits_calib=['']*n_idx

#sys.exit(0)
# i=2674
# 3C345-20190822@130653-R_Astrodon_2018.fts
# 3C345-20190714@135841-R_Astrodon_2018_calib.fits

# here is for test
ID_selected=np.array([1108,1109,1315,1316])   # 3C345
#idx_fitsheader=np.array([1107,1108,1314,1315])   # 3C345
#ID_selected=np.array([810,811,1334,1335])   # ES2344+514
#idx_fitsheader=np.array([809,810,1333,1334,1575,1576])   # ES2344+514
#ID_selected=np.array([814,815,1336,1337])  # L-Lacertae
idx_fitsheader=ID_selected-1  

JD=df_info['JD'][idx_fitsheader]


print(idx_fitsheader)
print(fits_ori)
print(fits_ori[idx_fitsheader])

n_idx=len(idx_fitsheader)
Rmag_targets=np.array([0.]*n_idx)
Rmag_err_targets=np.array([0.]*n_idx)
SNR_targets=np.array([0.]*n_idx)


#sys.exit(0)
'''
r_circle=10.
r_circle_as=r_circle*u.arcsec
r_inner=15.
r_outer=20.
r_inner_as=r_inner*u.arcsec
r_outer_as=r_outer*u.arcsec
aperture=SkyCircularAperture(positions, r_circle_as)
#print(aperture)
r_as=aperture.r
print('r_as =',r_as)
'''
#r_circle=8.
#print(r_circle,'pixel')
#r_circle_as=r_circle*u.arcsec
#aperture=SkyCircularAperture(positions, r_circle_as)

dx=300
dy=dx

j3=0


k=0
for i in idx_fitsheader:
    print('-----------------------')
    print('idx',i, ') ID =',ID[i],', #', k+1,'/',n_idx)
    fits_root=fits_ori[i].split('.',-1)[0].split('_calib',-1)[0]
    fits_calib=fits_root+'_calib.fits'
    print(fits_calib)
    #print(fits_root)
    #print(fits_calib)
    #print(fits_ori)
#   sys.exit(0)
#    print(radec_deg)
#    print(rmag)
    date=fits_root.split('@',-1)[0].split('-',-1)[-1]
    year=date[0:4]
    month=date[4:6]
    day=date[6:8]
    yearmonth=date[0:6]
#   sys.exit(0)
    dir_file=yearmonth+'/slt'+date+'_calib_sci/'
#    dir_reg=yearmonth+'/slt'+date+'_reg/'
    hdu=fits.open(dir_file+fits_calib)[0]
    imhead=hdu.header
    imdata=hdu.data
    wcs = WCS(imhead)
#    print(wcs)

    r_circle=fwhm[i]*xfwhm
    print('fwhm =',fwhm[i],', r =',r_circle,'pix')
    r_circle_as=r_circle*u.arcsec
    aperture=SkyCircularAperture(positions, r_circle_as)
    


#    radec_pix=wcs.all_world2pix(ra_deg,dec_deg,1)
    ra_pix,dec_pix=wcs.all_world2pix(ra_deg,dec_deg,1)
    ra_pix=ra_pix.tolist()
    dec_pix=dec_pix.tolist()
#    print(ra_pix,dec_pix)
    print()

#    j1=0
#    for j2 in idx_refstar:
#        ra_deg[j1]=df_refstar['RefStarRA_deg'][j2]
#        dec_deg[j1]=df_refstar['RefStarDEC_deg'][j2]
#        print(ra_deg[j1],dec_deg[j1])
#        radec_deg[j1]=[ra_deg[j1],dec_deg[j1]]
#        print(i,j1,radec_deg[j1])
#        ra_hhmmss[j1]=df_refstar['RefStarRA_hhmmss'][j2]
#        dec_ddmmss[j1]=df_refstar['RefStarDEC_ddmmss'][j2]
#        print(i,j1,radec_deg[j1],ra_hhmmss[j1],dec_ddmmss[j1])
#        rmag[j1]=df_refstar['Rmag'][j2]
#        rmag_err[j1]=df_refstar['Rmag_err'][j2]
#        radec_pix[j1]=wcs.all_world2pix(ra_deg[j1],dec_deg[j1],1)
#        ra_pix[j1]=radec_pix[j1][0].tolist()
#        dec_pix[j1]=radec_pix[j1][1].tolist()
#        print(j1,j2,radec_deg[j1],radec_pix[j1])
#        j1=j1+1


#    mask=np.array

#    fi=open('imhd.txt','w')
#    fi.write(imhead)
#    fi.close()
#    print('WCS(imhead)',wcs)
#    plt.subplot(projection=wcs) 
#    plt.imshow(hdu.data, origin='lower') 
#    plt.grid(color='white', ls='solid')
#    plt.show()

#    r_circle=fwhm[i]*5
#    r_circle=15.
#    r_circle_as=r_circle*u.arcsec
#   aperture=SkyCircularAperture(position, r=4. * u.arcsec)
#    print(r_circle_as)

#    r_inner=fwhm[i]*8
#    r_outer=fwhm[i]*10
#    r_inner=20.
#    r_outer=25.
#    r_inner_as=r_inner*u.arcsec
#    r_outer_as=r_outer*u.arcsec
#    print(r_inner_as,r_outer_as)

#   sys.exit(0)

#    background level and error
    bkg_mean,bkg_median,bkg_std=sigma_clipped_stats(imdata,sigma=3.)
#    print(mean,median,std)
#    1176.6646745343921 1176.0827349796416 39.1334108277639
    
#    np.argwhere(np.isnan(imdata))

#    time.sleep(1)
#    print(WCS.world_axis_physical_types)
#    aperture_pix=aperture.to_pixel(wcs)
#    print(aperture_pix)
#    positions_pix=aperture_pix.positions
#    r_pix=aperture_pix.r
#    print('r_pix =',r_pix)
    positions_pix=(ra_pix,dec_pix)
    positions_pix=np.transpose(positions_pix)
    aperture_pix=CircularAperture(positions_pix, r_circle)
    
    mask_circle=aperture_pix.to_mask(method='center')
    mask_data=mask_circle[0].multiply(imdata)
    
    
    err_imdata=bkg_std/bkg_median*imdata
#    phot_table = aperture_photometry(imdata, aperture,wcs=wcs)
    phot_table = aperture_photometry(imdata, aperture_pix,error=err_imdata)
#    print(phot_table)
#    print(phot_table.colnames)
#   print(phot_table['sky_center'])
#    print(phot_table['xcenter'])
#    print(phot_table['ycenter'])
    aper_sum=phot_table['aperture_sum']
    phot_table['aperture_sum'].info.format = '%.8g'  # for consistent table output
    phot_table['xcenter'].info.format = '%.8g'  # for consistent table output
    phot_table['ycenter'].info.format = '%.8g'  # for consistent table output
    phot_table['aperture_sum_err'].info.format = '%.8g'  # for consistent table output

    bkg_sum=aper_sum[-1]




#    aper_annu=SkyCircularAnnulus(positions,r_inner_as,r_outer_as)
#    print(aper_annu)
#    print(aper_annu.r_in)
#    print(aper_annu.r_out)
#    aper_annu_pix=aper_annu.to_pixel(wcs)
#    print(aper_annu_pix)
#    r_in_annu_pix=aper_annu_pix.r_in
#    r_out_annu_pix=aper_annu_pix.r_out
#    print(r_in_annu_pix,r_out_annu_pix)

#    apper=[aperture_pix,aper_annu_pix]
#    phot_annu_table = aperture_photometry(imdata, apper)
#    print(phot_annu_table)
#    print(phot_annu_table.colnames)
#    aper_annu_sum0=phot_annu_table['aperture_sum_0']
#   print(aper_annu_sum0)
#    aper_annu_sum1=phot_annu_table['aperture_sum_1']
#   print(aper_annu_sum1)
#    bkg_mean = phot_annu_table['aperture_sum_1'] / aper_annu_pix.area
#   print(bkg_mean)
    '''
    annulus_masks = aper_annu_pix.to_mask(method='center')
    bkg_median = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(imdata)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    bkg_median = np.array(bkg_median)
    '''
#    bkg_sum = bkg_median * aperture_pix.area
#    bkg_sum = bkg_mean * aperture_pix.area
#   print(bkg_sum)
    phot_table['bkg_sum']=bkg_sum
    phot_table['bkg_sum'].info.format = '%.8g'  # for consistent table output

#    final_sum = phot_annu_table['aperture_sum_0'] - bkg_sum
    final_sum=aper_sum-bkg_sum
#   print(final_sum)
#    phot_annu_table['residual_aperture_sum'] = final_sum
    phot_table['bg_subtract_sum'] = final_sum
    
#    phot_annu_table['xcenter'].info.format = '%.8g'  # for consistent table output
#    phot_annu_table['ycenter'].info.format = '%.8g'  # for consistent table output
#    phot_annu_table['aperture_sum_0'].info.format = '%.8g'  # for consistent table output
#    phot_annu_table['aperture_sum_1'].info.format = '%.8g'  # for consistent table output
#    phot_annu_table['residual_aperture_sum'].info.format = '%.8g'  # for consistent table output
#   print(phot_annu_table['residual_aperture_sum'])  
#    print(phot_annu_table)

    phot_table['bg_subtract_sum'].info.format = '%.8g'  # for consistent table output
    print(phot_table)

#   sys.exit(0)

    j=0
    rmag[0]=-9999.
    rmag_target=rmag[0]
    print()
    print('j final_sum         mag_instrument      Rmag')
    for j in range(n_refstar-1):
        mag_instrument[j]=-2.5*np.log10(final_sum[j])
        print(j, final_sum[j],mag_instrument[j],rmag[j])

    mag_instrument_refstar=mag_instrument[1:n_refstar-1]
    rmag_refstar=rmag[1:n_refstar-1]

    mag_instrument_source=mag_instrument[:n_refstar-1]
    rmag_source=rmag[:n_refstar-1]

#    no values    
#    mag_instrument_bkg=mag_instrument[-1]
#    rmag_bkg=rmag[-1]

    print('Instrument Magnitude of RefStars',mag_instrument_refstar)
    print('Rmag of RefStars',rmag_refstar)    
    print()
    b,m=polyfit(mag_instrument_refstar,rmag_refstar,1)
#   print('b =','%.3f' %b)
#   print('m =','%.3f' %m)
    text_fitting='Rmag = '+str('%.2f' %b)+' + '+str('%.2f' %m)+'*(Instrument Magnitude)'
    print(text_fitting)

    mag_instrument_target=mag_instrument[0]
    rmag_target=b+m*mag_instrument_target
#    print(rmag_target)

    Rmag_targets[k]=rmag_target

#    SN=signal_mean/signal_std
#    SN=bkg_mean/bkg_std ??    
#    Rmag_err=2.5*log10(1+N/S)=1.086/(S/N)
#    signal=final_sum
#    noise=bkg_std*aperture_pix.area
#    mag_err=1.086/snr

    
#    Rmag_err_targets[k]=mag_err[0]


#    plt.figure()
    mask_circle=aperture_pix.to_mask(method='center')
#    plt.imshow(mask_circle[3])
#    plt.colorbar()
#    plt.show()
    data_in_mask=mask_circle[3].multiply(imdata)
#    plt.imshow(data_in_mask)
#    plt.colorbar()
#    plt.show()
#    plt.close()
    
    
    mask_median = []
    for mask in mask_circle:
        circle_data = mask.multiply(imdata)
        circle_data_1d = circle_data[mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(circle_data_1d)
        mask_median.append(median_sigclip)
#        bkg_median = np.array(bkg_median)
    

    snr=mask_median/bkg_std
    mag_err=1.086/snr
    print('snr',snr)
    print('mag_err',mag_err)
    mag_err_target=mag_err[0]
    Rmag_err_targets[k]=mag_err[0]
    SNR_targets[k]=snr[0]

    phot = aperture_photometry(imdata, aperture_pix)
#    phot['circle_median'] = bkg_median
#    phot['bkg_median'] = bkg_median
    phot['aper_bkg'] = bkg_median * aperture_pix.area
#    phot['aper_bkg'] = bkg_sum
    phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
    for col in phot.colnames:
        phot[col].info.format = '%.8g'  # for consistent table output
    print(phot)
    
    

    figtitle=fits_ori[i]
    
    plt.figure(figsize=(8,6),dpi=120)
#    if change ra,dec of reference star, run 'gasp_refStar_radec.py again.    
    norm = simple_norm(imdata, 'sqrt', percent=99)
    plt.imshow(imdata, norm=norm)
    aperture_pix[0].plot(color='red', lw=2)
    aperture_pix[1:-1].plot(color='white', lw=2)
    aperture_pix[-1].plot(color='black', lw=2)
    x0=ra_pix[0]
    y0=dec_pix[0]
    x1=x0-dx
    x2=x0+dx
    y1=y0-dy
    y2=y0+dy
    plt.xlim(x1,x2)
    plt.ylim(y1,y2)
    plt.xlabel('RA (J2000)')
    plt.ylabel('Dec (J2000)')
    plt.title(figtitle)
    text_rmag='Rmag='+str('%.3f' %rmag_target)
    text_rmag_err='+/-'+str('%.3f' %mag_err_target)
    text_fwhm='fwhm='+str('%.3f' %fwhm[i])
    text_radius='r = fwhm x '+str(xfwhm)
#    text_fig=text_rmag+', '+text_fwhm
    plt.text(x0*0.95,y0*0.96,obj_name,fontsize=12,color='red')
    plt.text(x0*0.95,y0*0.93,text_rmag+text_rmag_err,fontsize=12)
    plt.text(x0*0.95,y0*0.90,text_fwhm,fontsize=12)
    plt.text(x0*0.95,y0*0.87,text_radius,fontsize=12)
#    plt.text(x2*0.55,y2*0.45,text_fwhm,fontsize=12)
#    plt.show()    
    plt.savefig(dir_obj+'Rmag_InsMag_aper_'+obj_name+'_'+date+'_fwhm'+str(xfwhm)+'_'+str(k)+'.png')
    plt.close()
    
    plt.figure()
#    plt.scatter(mag_instrument_refstar,rmag_refstar)
#    plt.plot(mag_instrument_refstar,rmag_refstar,'o')
    rmag_fitting_source=b+m*mag_instrument_source
    plt.plot(mag_instrument_source,rmag_fitting_source,'-',color='grey')
    plt.plot(mag_instrument_refstar,rmag_refstar,'o')
    plt.plot(mag_instrument_target,rmag_target,'o',color='red')
#    plt.errorbar(mag_instrument_source,rmag_source,yerr=mag_err[:-1])
    plt.xlabel('Instrument Magnitude')
    plt.ylabel('Rmag')
    plt.title(figtitle)
#    text_fig='Rmag='+str('%.3f' %rmag_target)+', fwhm='+str('%.3f' %fwhm[i])
    plt.text(mag_instrument_target-0.5,rmag_target-0.5,obj_name,fontsize=12,color='red')
    plt.text(mag_instrument_target-0.5,rmag_target-0.8,text_rmag,fontsize=12)
#    plt.text(mag_instrument_target-0.1,rmag_target-1.1,text_rmag_err,fontsize=12)
    plt.text(mag_instrument_target-0.5,rmag_target-1.1,text_fwhm,fontsize=12)
    plt.text(mag_instrument_target-0.5,rmag_target-1.4,text_radius,fontsize=12)

    imin=min(mag_instrument_source)
    imax=max(mag_instrument_source)
    rmin=min(rmag_fitting_source)
    rmax=max(rmag_fitting_source)
#    plt.text(imin+0.05,rmax-0.1,text_fitting,fontsize=12)
    plt.text(imin+0.3,rmin+0.1,text_fitting,fontsize=12)

#    plt.text(mag_instrument_target*0.9,rmag_target*0.9,'Rmag='+str(rmag_target))
#    plt.text(x0*0.85,y0*0.95,text_fig,fontsize=12)

#    plt.xlim(-14,-9)
#    plt.ylim(13,18)

#    print(mag_instrument,Rmag_targets)
#    plt.autoscale(enable=True,axis='both')
#    ax=plt.gca()
#    ax.relim()
#    ax.autoscale()
#    ax.Axes.autoscale #(enable=True,axis='both')
#    ymin=min(Rmag_targets)
#    ymax=max(Rmag_targets)
#    plt.ylim(ymin,ymax)
#    xmin=min(mag_instrument)
#    xmax=max(mag_instrument)
#    plt.xlim(xmin,xmax)
    
#    set_xlim(auto=True)
#    set_ylim(auto=True)
#    plt.show()
    plt.savefig(dir_obj+'Rmag_InsMag_rmag_'+obj_name+'_'+date+'_fwhm'+str(xfwhm)+'_'+str(k)+'.png')
    plt.close()    
    print('target new Rmag =',rmag_target)
#    print('Instrument Magnitude',mag_instrument)


    k=k+1
    

print('-----------------------')
print('Rmag',Rmag_targets)



df_out=df_info.iloc[idx_fitsheader]
df_out['RA_deg']=df_out['RA_deg'].map('{:,.4f}'.format)
df_out['DEC_deg']=df_out['DEC_deg'].map('{:,.4f}'.format)
df_out['RA_pix']=df_out['RA_pix'].map('{:,.4f}'.format)
df_out['Dec_pix']=df_out['Dec_pix'].map('{:,.4f}'.format)
df_out['Zmag']=df_out['Zmag'].map('{:,.4f}'.format)
df_out['FWHM']=df_out['FWHM'].map('{:,.4f}'.format)
df_out['Altitude']=df_out['Altitude'].map('{:,.4f}'.format)
df_out['Airmass']=df_out['Airmass'].map('{:,.4f}'.format)

df_out['Rmag']=Rmag_targets
df_out['Rmag']=df_out['Rmag'].map('{:,.4f}'.format)
df_out['Rmag_err']=Rmag_err_targets
df_out['Rmag_err']=df_out['Rmag_err'].map('{:,.4f}'.format)
df_out['SNR']=SNR_targets
df_out['SNR']=df_out['SNR'].map('{:,.4f}'.format)


print(df_out)
file_Rmag_out=dir_obj+'Rmag_aperture_'+obj_name+'_fwhm'+str(xfwhm)+'.txt'
df_out.to_csv(file_Rmag_out,sep='|')


plt.figure()
#plt.scatter(JD,Rmag_targets,label='data')
plt.errorbar(JD,Rmag_targets,yerr=Rmag_err_targets,label='data',fmt='o')
#plt.plot(JD,Rmag_targets,'-')
plt.xlabel('JD')
plt.ylabel('Rmag')
plt.gca().invert_yaxis()
plt.title(obj_name+' | fwhm x'+str(xfwhm))
plt.legend(loc='best')
#plt.show()
plt.savefig(dir_obj+'Rmag_JD_'+obj_name+'_fwhm'+str(xfwhm)+'.png')
plt.close()

plt.figure()
plt.plot(JD,Rmag_targets,'-',label='lightcurve',color='grey')
#plt.scatter(JD,Rmag_targets,label='data')
plt.errorbar(JD,Rmag_targets,yerr=Rmag_err_targets,label='data',fmt='o')

plt.xlabel('JD')
plt.ylabel('Rmag')
plt.gca().invert_yaxis()
plt.title(obj_name+' | fwhm x'+str(xfwhm))
plt.legend(loc='best')
#plt.show()
plt.savefig(dir_obj+'Rmag_JD_'+obj_name+'_fwhm'+str(xfwhm)+'_line.png')
plt.close()
#sys.exit(0)

rmag_nan=([])

k=0
for i in idx_fitsheader:
    ID_fitsheader=ID[i]
    if Rmag_targets[k]==np.nan:
        rmag_nan=np.append(rmag_nan,ID_fitsheader)
    k=k+1

print('Rmag with nan value:',rmag_nan)

print('... write file to: ./'+file_Rmag_out)
print('... finished ...')
#==============================


'''  
file_reg_fk5=dir_refstar+'RefStar_'+obj_name+'_annu_fk5.reg'
print('will write to : '+file_reg_fk5)
if os.path.exists(file_reg_fk5):
    os.remove(file_reg_fk5)
f_reg=open(file_reg_fk5,'w')



f_reg.write('# Region file format: DS9 version 4.1\n')
f_reg.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
f_reg.write('fk5\n')

k=0
for i in idx_refstar:
    if k==0:
        txt_target='circle('+str(ra_hhmmss[k])+','+str(dec_ddmmss[k])+','+str(r_circle)+'") # color=red'
        txt_bkg='annulus('+str(ra_hhmmss[k])+','+str(dec_ddmmss[k])+','+str(r_inner)+'",'+str(r_outer)+'")'
    else:
        txt_target='circle('+str(ra_hhmmss[k])+','+str(dec_ddmmss[k])+','+str(r_circle)+'") # color=white'
        txt_bkg='annulus('+str(ra_hhmmss[k])+','+str(dec_ddmmss[k])+','+str(r_inner)+'",'+str(r_outer)+'")'
#    print(k,i)
    f_reg.write(txt_target+'\n')
    f_reg.write(txt_bkg+'\n')
    k=k+1
f_reg.close()
'''

'''
#img='62_z_CDFs_goods_stamp_img.fits'  #path to the image
#RA = 52.9898239
#DEC = -27.7143114
#hdulist = astropy.io.fits.open(img)
#w = wcs.WCS(hdulist['PRIMARY'].header)
#world = np.array([[RA, DEC]])
pix = wcs.wcs_world2pix(radec_deg,1) # Pixel coordinates of (RA, DEC)
print( "Pixel Coordinates: ", pix[0,0], pix[0,1])
sys.exit(0)
#call aperture function
observation=aperphot(imdata, timekey=None, pos=[pix[0,0], pix[0,1]], dap=[4*fwhm,8*fwhm,12*fwhm], resamp=2, retfull=False)

# Print outputs
print( "Aperture flux:", observation.phot)
print( "Background:   ", observation.bg)
'''


'''
annulus_masks = aper_annu_pix.to_mask(method='center')
plt.imshow(annulus_masks)
plt.colorbar()

annulus_data = annulus_masks[0].multiply(imdata)
plt.imshow(annulus_data)
plt.colorbar()

mask = annulus_masks[0].data
annulus_data_1d = annulus_data[mask > 0]
annulus_data_1d.shape
'''



'''
annulus_aperture = CircularAnnulus(position, r_in=fwhm*6.* u.arcsec, r_out=fwhm*8.* u.arcsec)
apers=[aperture,annulus_aperture]
phot_table2=aperture_photometry(imdata,apers)
for col in phot_table2.colnames:
    phot_table2[col].info.format = '%.8g'  # for consistent table output
(phot_table2)

#bkg_mean = phot_table2['aperture_sum_1'] / annulus_aperture.area
'''
