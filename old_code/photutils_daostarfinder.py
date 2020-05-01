#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:40:28 2019

@author: altsai
"""


import os
import sys
import numpy as np
import csv
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from astropy.table import Table
from photutils import CircularAperture
from photutils import SkyCircularAperture
from photutils import aperture_photometry
from photutils import CircularAnnulus
# https://photutils.readthedocs.io/en/stable/aperture.html
from phot import aperphot
# http://www.mit.edu/~iancross/python/phot.html

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from photutils import DAOStarFinder
from astropy.stats import mad_std
# https://photutils.readthedocs.io/en/stable/getting_started.html

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

file_info='gasp_target_fitsheader_info_slt'+date+'.txt'
#file_info='gasp_target_fitsheader_info_slt'+year+month+'.txt'
print('... will read '+file_info+' ...')
df_info=pd.read_csv(file_info,delimiter='|')
#print(df_info)

idx_fitsheader=df_info[df_info['Filename']==fits_ori].index[0]
print(idx_fitsheader)
obj_name=df_info['Object'][idx_fitsheader]
fwhm=df_info['FWHM'][idx_fitsheader]
print(fwhm)

ra0_deg=df_info['RA_deg'][idx_fitsheader]
dec0_deg=df_info['DEC_deg'][idx_fitsheader]


dir_refstar='./RefStar/'
file_refstar='gasp_refStar_radec.txt'

df_refstar=pd.read_csv(file_refstar,sep='|')
#print(df_refstar)
idx_refstar=df_refstar[df_refstar['ObjectName']==obj_name].index.tolist()
n_refstar=len(idx_refstar)
ra_deg=[0]*n_refstar
dec_deg=[0]*n_refstar
radec_deg=[[0,0]]*n_refstar
j=0
for i in idx_refstar:
    ra_deg[j]=df_refstar['RefStarRA_deg'][i]
    dec_deg[j]=df_refstar['RefStarDEC_deg'][i]
#    print(ra_deg[j],dec_deg[j])
    radec_deg[j]=[ra_deg[j],dec_deg[j]]
#    print(radec_deg[j])
    j=j+1

radec_deg=np.array(radec_deg)
print(radec_deg)

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

bkg_sigma = mad_std(imdata)  
daofind = DAOStarFinder(fwhm=4.*fwhm, threshold=3.*bkg_sigma)  
sources = daofind(imdata)  
for col in sources.colnames:
    sources[col].info.format = '%.8g'  # for consistent table output
print(sources)  
print(sources.colnames)

positions = np.transpose((sources['xcentroid'], sources['ycentroid']))  
apertures = CircularAperture(positions, r=4.)  
phot_table = aperture_photometry(imdata, apertures)  
for col in phot_table.colnames:  
    phot_table[col].info.format = '%.8g'  # for consistent table output
print(phot_table)  
print(phot_table.colnames)

plt.imshow(imdata, cmap='gray_r', origin='lower')
apertures.plot(color='blue', lw=1.5, alpha=0.5)
