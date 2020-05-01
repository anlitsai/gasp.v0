#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:42:51 2019

@author: altsai
"""


import os
import sys
import shutil
#import re
import numpy as np
#import numpy
from astropy.io import fits
#import matplotlib.pyplot as plt
#import scipy.ndimage as snd
#import glob
#import subprocess
#from scipy import interpolate
#from scipy import stats
#from scipy.interpolate import griddata
#from time import gmtime, strftime
#import pandas as pd
from datetime import datetime
#from scipy import interpolate
#from scipy import stats
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii


print('... will read target information for a month ...')
month=input("Enter a year-month (ex: 201908): ")
file_info='gasp_target_fitsheader_info_slt'+month+'.txt'
print('... will read '+file_info+' ...')
print()
f_info=ascii.read(file_info) #,delimiter='|')
head_column=f_info.colnames
print(head_column)

column_idx=f_info['Index']
column_obj=f_info['Object']
column_ra_hhmmss=f_info['RA(hhmmss)']
column_dec_ddmmss=f_info['DEC(ddmmss)']

print(column_obj)
print(column_idx[-1])


