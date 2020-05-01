#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 00:11:00 2019

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
from astropy.visualization import simple_norm

dir_snapshot='./bad_data/snapshot/'
if not os.path.exists(dir_snapshot):
    os.makedirs(dir_snapshot,exist_ok=True)

#cmd_search_baddata="find ./ |grep bad_data|grep fts|sort"
cmd_search_baddata="find ./bad_data/|grep fts|sort"
print(cmd_search_baddata)
list_baddata=os.popen(cmd_search_baddata,"r").read().splitlines()
print(list_baddata)


for i in list_baddata:
    filename=str(i.split('/',-1)[-1].split('.',-1)[0])+'.png'
    print(filename)
    hdu=fits.open(i)[0]
    imdata=hdu.data
    norm = simple_norm(imdata, 'sqrt', percent=99.5)
    plt.imshow(imdata, norm=norm)
    plt.savefig(dir_snapshot+filename)

plt.close()