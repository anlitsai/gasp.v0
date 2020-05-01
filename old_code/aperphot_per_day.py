#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 05:22:35 2019

@author: altsai
"""

import os
import sys
import numpy as np
import csv
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5 # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy.coordinates import match_coordinates_sky
from astropy.table import Table
from photutils import CircularAperture
from photutils import SkyCircularAperture
from photutils import aperture_photometry
from photutils import CircularAnnulus
from photutils import SkyCircularAnnulus
# https://photutils.readthedocs.io/en/stable/aperture.html
from phot import aperphot
# http://www.mit.edu/~iancross/python/phot.html

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from photutils import DAOStarFinder
from astropy.stats import mad_std
# https://photutils.readthedocs.io/en/stable/getting_started.html

from numpy.polynomial.polynomial import polyfit



date=input('Please input the date to do apereture photometry: ')
'''
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
'''
year=date[0:4]
month=date[4:6]
day=date[6:8]

file_info='gasp_target_fitsheader_info_slt'+year+'.txt'
#file_info='gasp_target_fitsheader_info_slt'+date+'.txt'
#file_info='gasp_target_fitsheader_info_slt'+year+month+'.txt'
print('... will read '+file_info+' ...')
df_info=pd.read_csv(file_info,delimiter='|')
#print(df_info)


dateobs=year+'-'+month+'-'+day
idx_date=df_info[df_info['DateObs']==dateobs].index
print(idx_date)
'''
idx_fitsheader=df_info[df_info['Filename']==fits_ori].index[0]
print(idx_fitsheader)
obj_name=df_info['Object'][idx_fitsheader]
fwhm=df_info['FWHM'][idx_fitsheader]
print(fwhm)

ra0_deg=df_info['RA_deg'][idx_fitsheader]
dec0_deg=df_info['DEC_deg'][idx_fitsheader]
'''
n_date=len(idx_date)
obj_name=np.array(['']*n_date)
fwhm=np.array([0.]*n_date)
k=0
for i in idx_date:
    obj_name[k]=df_info['Object'][idx_date][k]
    fwhm[k]=df_info['FWHM'][idx_date][k]
    print(fwhm[k])




#dir_refstar='./RefStar/'
file_refstar='gasp_refStar_radec.txt'

df_refstar=pd.read_csv(file_refstar,sep='|')
#print(df_refstar)
idx_refstar=df_refstar[df_refstar['ObjectName']==obj_name].index.tolist()
n_refstar=len(idx_refstar)
ra_deg=np.array([0.]*n_refstar)
dec_deg=np.array([0.]*n_refstar)
radec_deg=np.array([[0.,0.]]*n_refstar)
rmag=np.array([0.]*n_refstar)
rmag_err=np.array([0.]*n_refstar)

j=0
for i in idx_refstar:
    ra_deg[j]=df_refstar['RefStarRA_deg'][i]
    dec_deg[j]=df_refstar['RefStarDEC_deg'][i]
#    print(ra_deg[j],dec_deg[j])
    radec_deg[j]=[ra_deg[j],dec_deg[j]]
#    print(radec_deg[j])
    rmag[j]=df_refstar['Rmag'][i]
    rmag_err[j]=df_refstar['Rmag_err'][i]
    j=j+1

#radec_deg=np.array(radec_deg)
print(radec_deg)
print(rmag)
#sys.exit(0)

#pix_aperture = aperture.to_pixel(wcs)
#sky_aperture = aperture.to_sky(wcs)
#import pywcs
#import pyfits

hdu=fits.open(fits_calib)[0]
imhead=hdu.header
imdata=hdu.data
wcs = WCS(imhead)
print('WCS(imhead)',wcs)
plt.subplot(projection=wcs) 
plt.imshow(hdu.data, origin='lower') 
plt.grid(color='white', ls='solid')
plt.show()
#sys.exit(0)


#position= SkyCoord([ICRS(ra=ra_deg*u.deg,dec=dec_deg*u.deg)])
position= SkyCoord(ra_deg,dec_deg,unit=(u.hourangle,u.deg),frame='icrs')
print(position)
print(position.ra)
print(position.dec)
#sys.exit(0)
r4=fwhm*4. * u.arcsec
r5=fwhm*5. * u.arcsec
r6=fwhm*6. * u.arcsec
r7=fwhm*7. * u.arcsec


#aperture=SkyCircularAperture(position, r=4. * u.arcsec)
aperture=SkyCircularAperture(position, r4)
print(aperture)
r_as=aperture.r
print(r_as)

#sys.exit(0)
aperture_pix=aperture.to_pixel(wcs)
print(aperture_pix)
r_pix=aperture_pix.r
print(r_pix)

#sys.exit(0)

#phot_table = aperture_photometry(imdata, aperture,wcs=wcs)
phot_table = aperture_photometry(imdata, aperture_pix)
print(phot_table)
print(phot_table.colnames)

#print(phot_table['sky_center'])
print(phot_table['xcenter'])
print(phot_table['ycenter'])
aper_sum=phot_table['aperture_sum']
phot_table['aperture_sum'].info.format = '%.8g'  # for consistent table output




aper_annu=SkyCircularAnnulus(position,r5,r6)
print(aper_annu)
print(aper_annu.r_in)
print(aper_annu.r_out)
aper_annu_pix=aper_annu.to_pixel(wcs)
print(aper_annu_pix)
r_in_annu_pix=aper_annu_pix.r_in
r_out_annu_pix=aper_annu_pix.r_out
print(r_in_annu_pix,r_out_annu_pix)

apper=[aperture_pix,aper_annu_pix]
phot_annu_table = aperture_photometry(imdata, apper)
print(phot_annu_table)
print(phot_annu_table.colnames)
aper_annu_sum0=phot_annu_table['aperture_sum_0']
#print(aper_annu_sum0)
aper_annu_sum1=phot_annu_table['aperture_sum_1']
#print(aper_annu_sum1)
bkg_mean = phot_annu_table['aperture_sum_1'] / aper_annu_pix.area
#print(bkg_mean)
bkg_sum = bkg_mean * aper_annu_pix.area
#print(bkg_sum)
final_sum = phot_annu_table['aperture_sum_0'] - bkg_sum
#print(final_sum)
phot_annu_table['residual_aperture_sum'] = final_sum

phot_annu_table['xcenter'].info.format = '%.8g'  # for consistent table output
phot_annu_table['ycenter'].info.format = '%.8g'  # for consistent table output
phot_annu_table['aperture_sum_0'].info.format = '%.8g'  # for consistent table output
phot_annu_table['aperture_sum_1'].info.format = '%.8g'  # for consistent table output
phot_annu_table['residual_aperture_sum'].info.format = '%.8g'  # for consistent table output
#print(phot_annu_table['residual_aperture_sum'])  
print(phot_annu_table)
#sys.exit(0)

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


mag_instrument=np.array([0.]*n_refstar)
for i in range(n_refstar):
    mag_instrument[i]=-2.5*np.log10(final_sum[i])
    print(final_sum[i],mag_instrument[i],rmag[i])

mag_instrument_1=mag_instrument[1:n_refstar]
rmag_1=rmag[1:n_refstar]

b,m=polyfit(mag_instrument_1,rmag_1,1)
#print('b =','%.3f' %b)
#print('m =','%.3f' %m)
print('Rmag =','%.3f' %b,'+','%.3f' %m,'*(Instrument Magnitude)')

rmag[0]=b+m*mag_instrument[0]
print(rmag[0])

#plt.scatter(mag_instrument_1,rmag_1)
plt.plot(mag_instrument_1,rmag_1,'o')
plt.plot(mag_instrument[0],rmag[0],'o')
plt.plot(mag_instrument,b+m*mag_instrument,'-')
plt.xlabel('Instrument Magnitude')
plt.ylabel('Rmag')
plt.show()

print('Rmag',rmag)
print('Instrument Magnitude',mag_instrument)

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
annulus_aperture = CircularAnnulus(position, r_in=fwhm*6.* u.arcsec, r_out=fwhm*8.* u.arcsec)
apers=[aperture,annulus_aperture]
phot_table2=aperture_photometry(imdata,apers)
for col in phot_table2.colnames:
    phot_table2[col].info.format = '%.8g'  # for consistent table output
(phot_table2)

#bkg_mean = phot_table2['aperture_sum_1'] / annulus_aperture.area
'''