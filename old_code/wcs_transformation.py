#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:53:47 2019

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
#from astropy.wcs import WCS
from astropy import wcs
#from photutils import DAOStarFinder
#from astropy.stats import mad_std
# https://photutils.readthedocs.io/en/stable/getting_started.html

from numpy.polynomial.polynomial import polyfit
from astropy.stats import sigma_clipped_stats
from photutils.psf import IterativelySubtractedPSFPhotometry
from statistics import mode
from astropy.visualization import simple_norm
from photutils.utils import calc_total_error


file='./no_wcs/3C345-20190714@140156-R_Astrodon_2018_calib.fits'
#file='./3C345-20190822@130653-R_Astrodon_2018_calib.fits'
print(file)
print()

hdu=fits.open(file)[0]
imhead=hdu.header
imdata=hdu.data
w=wcs.WCS(naxis=2)
#w=WCS(imhead)
print(w)
print(w.wcs.name)
w.wcs.print_contents()
w.wcs.crval
print()
w.wcs.ctype=["RA-TAN","DEC--TAN"]
w.wcs.crpix=[1024,1024]   # center pixel
#w.wcs.cdelt
#w.wcs.cdelt = np.array([-0.066667, 0.066667])
#w.wcs.crval = [0, -90]
#w.wcs.set_pv([(2, 1, 45.0)])
#sys.exit(0)

ra_hhmmss=imhead['RA']
dec_ddmmss=imhead['DEC']
radec_deg= SkyCoord(ra_hhmmss,dec_ddmmss,unit=(u.hourangle,u.deg),frame='icrs')
print(radec_deg)
glat=np.array(radec_deg.ra.deg)
glon=np.array(radec_deg.dec.deg)
#glat=imhead['LAT-OBS']
#glon=imhead['LONG-OBS']
airmass=imhead['AIRMASS']
xbins=imhead['NAXIS1']
ybins=imhead['NAXIS2']
#cdeltX=2.19316e-4
cdeltX=2.193e-4  # +/- unknown
cdeltY=2.193e-4  # +/- unknown
Av=1/airmass
print(glat,glon,xbins,ybins)

xbins=np.array([2048])
ybins=np.array([2048])
#xbins=100
#ybins=100
H, xedges, yedges = np.histogram2d(glat, glon, bins=[ybins, xbins], weights=Av)
count, x, y = np.histogram2d(glat, glon, bins=[ybins, xbins])
H/=count

# characterize your data in terms of a linear translation from XY pixels to 
# Galactic longitude, latitude. 

# lambda function given min, max, n_pixels, return spacing, middle value.
linwcs = lambda x, y, n: ((x-y)/n, (x+y)/2)

cdeltaX, crvalX = linwcs(np.amin(glon), np.amax(glon), len(glon))
cdeltaY, crvalY = linwcs(np.amin(glat), np.amax(glat), len(glat))

# wcs code ripped from 
# http://docs.astropy.org/en/latest/wcs/index.html

w = wcs.WCS(naxis=2)

# what is the center pixel of the XY grid.
w.wcs.crpix = [len(glon)/2, len(glat)/2]

# what is the galactic coordinate of that pixel.
w.wcs.crval = [crvalX, crvalY]

# what is the pixel scale in lon, lat.
w.wcs.cdelt = np.array([cdeltX, cdeltY])

# you would have to determine if this is in fact a tangential projection. 
w.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]

print(w)
'''
# write the HDU object WITH THE HEADER
header = w.to_header()
hdu = pyfits.PrimaryHDU(H, header=header)
hdu.writeto(filename)
'''