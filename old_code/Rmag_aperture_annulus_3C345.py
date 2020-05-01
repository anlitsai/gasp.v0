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
from astropy.io import fits
from astropy.wcs import WCS
#from photutils import DAOStarFinder
#from astropy.stats import mad_std
# https://photutils.readthedocs.io/en/stable/getting_started.html

from numpy.polynomial.polynomial import polyfit


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
file_info='testheader.txt'
#file_info='gasp_target_fitsheader_info_slt2019.txt'
#file_info='gasp_target_fitsheader_info_slt'+date+'.txt'
#file_info='gasp_target_fitsheader_info_slt'+year+month+'.txt'
print('... will read '+file_info+' ...')
df_info=pd.read_csv(file_info,delimiter='|')
#print(df_info)

obj_name='3C345'

dir_obj='Rmag_InstMag/'+obj_name+'/annu/'
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
radec_deg=np.array([[0.,0.]]*n_refstar)
ra_hhmmss=['']*n_refstar
dec_ddmmss=['']*n_refstar

rmag=np.array([0.]*n_refstar)
rmag_err=np.array([0.]*n_refstar)
print(rmag)
mag_instrument=np.array([0.]*n_refstar)


j=0
for i in idx_refstar:
    ra_deg[j]=df_refstar['RefStarRA_deg'][i]
    dec_deg[j]=df_refstar['RefStarDEC_deg'][i]
#    print(ra_deg[j],dec_deg[j])
    radec_deg[j]=[ra_deg[j],dec_deg[j]]
#    print(i,j,radec_deg[j])
    ra_hhmmss[j]=df_refstar['RefStarRA_hhmmss'][i]
    dec_ddmmss[j]=df_refstar['RefStarDEC_ddmmss'][i]
#    print(i,j,radec_deg[j],ra_hhmmss[j],dec_ddmmss[j])
    rmag[j]=df_refstar['Rmag'][i]
    rmag_err[j]=df_refstar['Rmag_err'][i]
    j=j+1





idx_fitsheader=df_info[df_info['Object']==obj_name].index
#print(idx_fitsheader)

#obj_name=df_info['Object'][idx_fitsheader]
fwhm=df_info['FWHM'][idx_fitsheader]
#print(fwhm)
ID=df_info['ID'][idx_fitsheader]

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
#idx_fitsheader=np.array([1107,1315,1316])
#idx_fitsheader=np.array([1107,1108,1315,1316])
#idx_fitsheader=np.array(int(input('1107,1108,1315,1316 ? ')))
#idx_fitsheader=np.array([1107])
#idx_fitsheader=np.array([1108])
#idx_fitsheader=np.array([1109])
#idx_fitsheader=np.array([2673])
print(idx_fitsheader)
print(fits_ori)
print(fits_ori[idx_fitsheader])

n_idx=len(idx_fitsheader)
Rmag0=np.array([0.]*n_idx)
#sys.exit(0)

r_circle=15.
r_circle_as=r_circle*u.arcsec
r_inner=20.
r_outer=25.
r_inner_as=r_inner*u.arcsec
r_outer_as=r_outer*u.arcsec
aperture=SkyCircularAperture(positions, r_circle_as)
#print(aperture)
r_as=aperture.r
print('r_as =',r_as)


k=0
for i in idx_fitsheader:
    print('-----------------------')
    print('idx',i, ') ID =',ID[i],', #', k)
    fits_root=fits_ori[i].split('.',-1)[0].split('_calib',-1)[0]
    fits_calib=fits_root+'_calib.fits'
    print(fits_calib)
    #print(fits_root)
    #print(fits_calib)
    #print(fits_ori)
#   sys.exit(0)
#    print(radec_deg)
#    print(rmag)
    date=fits_root.split('@',-1)[0].split('-',-1)[1]
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

#    time.sleep(1)
#    print(WCS.world_axis_physical_types)
    aperture_pix=aperture.to_pixel(wcs)
#    print(aperture_pix)
    r_pix=aperture_pix.r
    print('r_pix =',r_pix)
#    phot_table = aperture_photometry(imdata, aperture,wcs=wcs)
    phot_table = aperture_photometry(imdata, aperture_pix)
#    print(phot_table)
#    print(phot_table.colnames)
#   print(phot_table['sky_center'])
#    print(phot_table['xcenter'])
#    print(phot_table['ycenter'])
    aper_sum=phot_table['aperture_sum']
    phot_table['aperture_sum'].info.format = '%.8g'  # for consistent table output

    aper_annu=SkyCircularAnnulus(positions,r_inner_as,r_outer_as)
#    print(aper_annu)
#    print(aper_annu.r_in)
#    print(aper_annu.r_out)
    aper_annu_pix=aper_annu.to_pixel(wcs)
#    print(aper_annu_pix)
    r_in_annu_pix=aper_annu_pix.r_in
    r_out_annu_pix=aper_annu_pix.r_out
#    print(r_in_annu_pix,r_out_annu_pix)

    apper=[aperture_pix,aper_annu_pix]
    phot_annu_table = aperture_photometry(imdata, apper)
#    print(phot_annu_table)
#    print(phot_annu_table.colnames)
    aper_annu_sum0=phot_annu_table['aperture_sum_0']
#   print(aper_annu_sum0)
    aper_annu_sum1=phot_annu_table['aperture_sum_1']
#   print(aper_annu_sum1)
    bkg_mean = phot_annu_table['aperture_sum_1'] / aper_annu_pix.area
#   print(bkg_mean)
    bkg_sum = bkg_mean * aperture_pix.area
#   print(bkg_sum)
    final_sum = phot_annu_table['aperture_sum_0'] - bkg_sum
#   print(final_sum)
    phot_annu_table['residual_aperture_sum'] = final_sum

    phot_annu_table['xcenter'].info.format = '%.8g'  # for consistent table output
    phot_annu_table['ycenter'].info.format = '%.8g'  # for consistent table output
    phot_annu_table['aperture_sum_0'].info.format = '%.8g'  # for consistent table output
    phot_annu_table['aperture_sum_1'].info.format = '%.8g'  # for consistent table output
    phot_annu_table['residual_aperture_sum'].info.format = '%.8g'  # for consistent table output
#   print(phot_annu_table['residual_aperture_sum'])  
#    print(phot_annu_table)
#   sys.exit(0)

    j=0
    rmag[0]=-9999.
    print('j final_sum         mag_instrument      Rmag[0]')
    for j in range(n_refstar):
        mag_instrument[j]=-2.5*np.log10(final_sum[j])
        print(j, final_sum[j],mag_instrument[j],rmag[j])

    mag_instrument_1=mag_instrument[1:n_refstar]
    rmag_1=rmag[1:n_refstar]

    b,m=polyfit(mag_instrument_1,rmag_1,1)
#   print('b =','%.3f' %b)
#   print('m =','%.3f' %m)
#    print('Rmag =','%.3f' %b,'+','%.3f' %m,'*(Instrument Magnitude)')

    rmag[0]=b+m*mag_instrument[0]
#    print(rmag[0])
    Rmag0[k]=rmag[0]

    plt.figure()
#   plt.scatter(mag_instrument_1,rmag_1)
    plt.plot(mag_instrument_1,rmag_1,'o')
    plt.plot(mag_instrument[0],rmag[0],'o')
    plt.plot(mag_instrument,b+m*mag_instrument,'-')
    plt.xlabel('Instrument Magnitude')
    plt.ylabel('Rmag')
    plt.title(fits_ori[i])
#    plt.show()
    plt.savefig(dir_obj+'Rmag_InsMag_'+obj_name+'_'+date+'_'+str(k)+'.png')
    plt.close()
    print('new Rmag[0] =',rmag[0])
#    print('Instrument Magnitude',mag_instrument)
    k=k+1
    

print('-----------------------')
print('Rmag',Rmag0)

#sys.exit(0)
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
